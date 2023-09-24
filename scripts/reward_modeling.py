# Reference: https://github.com/huggingface/trl/blob/main/examples/scripts/reward_trainer.py

from dataclasses import dataclass, field
from typing import Optional
import os
import json
from time import gmtime, strftime

import torch.distributed as dist
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
)

from functools import partial
from trl import RewardTrainer
import pdb

tqdm.pandas()

def print_rank_0(*args, **kwargs) -> None:
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)

def preprocess_function(examples, tokenizer, max_length):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
        tokenized_chosen = tokenizer(chosen)
        tokenized_rejected = tokenizer(rejected)
        if len(tokenized_chosen["input_ids"]) <= max_length and len(tokenized_rejected["input_ids"]) <= max_length:
            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    return new_examples

def prepare_exp_name(args):
    exp_name = "rm"
    exp_name += f"-lr{args.learning_rate}"
    exp_name += f"-bs{args.per_device_batch_size}"
    exp_name += f"-nsteps{args.max_steps}"
    exp_name += f"-r{args.lora_rank}"
    return exp_name

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with RewardTrainer
    """

    model_name: Optional[str] = field(default="yahma/llama-7b-hf", metadata={"help": "the model name"})
    data_path: Optional[str] = field(default="./data/ILF-refinement-rm", metadata={"help": "the model name"})
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    log_with: Optional[str] = field(default="wandb", metadata={"help": "use 'wandb' to log with wandb"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the number of update steps between two logs"})
    learning_rate: Optional[float] = field(default=1e-4, metadata={"help": "the learning rate"})
    per_device_batch_size: Optional[int] = field(default=1, metadata={"help": "the batch size"})
    num_train_epochs: Optional[int] = field(default=1, metadata={"help": "the number of training epochs"})
    seq_length: Optional[int] = field(default=1024, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    load_in_8bit: Optional[bool] = field(default=True, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=True, metadata={"help": "Wether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=True, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(default="./out/rm", metadata={"help": "the output directory"})
    max_steps: Optional[int] = field(default=1000, metadata={"help": "the maximum number of training steps"})
    debug: Optional[bool] = field(default=False, metadata={"help": "debug mode"})
    save_freq: Optional[int] = field(default=100, metadata={"help": "the frequency at which to save the model"})
    eval_freq: Optional[int] = field(default=100, metadata={"help": "the frequency at which to evaluate the model"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    num_warmup_steps: Optional[int] = field(default=10, metadata={"help": "the number of warmup steps"})
    no_fp16: Optional[bool] = field(default=False, metadata={"help": "disable fp16"})
    bf16: Optional[bool] = field(default=False, metadata={"help": "use bf16"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    exp_name: Optional[str] = field(default="exp", metadata={"help": "the name of the experiment"})
    padding_side: Optional[str] = field(default="left", metadata={"help": "the padding side"})
    train_pct: Optional[float] = field(default=0.5, metadata={"help": "the percentage of the training set to use"})
    lora_rank: Optional[int] = field(default=8, metadata={"help": "the rank of the Lora matrix"})

def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    if not script_args.debug:
        date_time = strftime("%Y-%m-%d-%H:%M", gmtime())
        script_args.exp_name = prepare_exp_name(script_args)
        script_args.output_dir = os.path.join(script_args.output_dir, "sft", date_time+"-"+script_args.exp_name) # BUG: no need to add sft subfolder
        try:
            os.makedirs(script_args.output_dir)
        except FileExistsError:
            pass
        
        with open(os.path.join(script_args.output_dir, "args.json"), "w") as f:
            json.dump(script_args.__dict__, f, indent=4)
    else:
        script_args.exp_name = "debug-session"
        script_args.max_steps = 10

    # Step 1: Load the model
    if script_args.load_in_8bit and script_args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif script_args.load_in_8bit or script_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
        )
        # This means: fit the entire model on the GPU:0
        device_map = {"": Accelerator().process_index}
    else:
        device_map = {"": Accelerator().process_index}
        quantization_config = None

    model = AutoModelForSequenceClassification.from_pretrained(
        script_args.model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=script_args.trust_remote_code,
        num_labels=1,
    )

    # Step 2: Load the dataset and pre-process it
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, add_eos_token=True)
    tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = script_args.padding_side

    if script_args.data_path == "./data/ILF-refinement-rm":
        dataset_files = {
            "train": os.path.join(script_args.data_path, "train.jsonl"),
            "validation": os.path.join(script_args.data_path, "validation.jsonl")
        }
        data = load_dataset("json", data_files=dataset_files)

    elif script_args.data_path == "Anthropic/hh-rlhf":
        data = load_dataset(script_args.data_path, data_dir="helpful-base") # Using only a subset
        data["validation"] = data["test"]
        del data["test"]
    train_data, eval_data = data["train"].select(range(int(script_args.train_pct * len(data["train"])))), data["validation"]

    if script_args.debug:
        train_data, eval_data = train_data.select(range(50)), eval_data.select(range(20))

    preprocess_partial = partial(preprocess_function, tokenizer=tokenizer, max_length=script_args.seq_length)
    train_data = train_data.map(
        preprocess_partial,
        batched=True,
        num_proc=4,
        remove_columns=train_data.column_names
    )

    eval_data = eval_data.map(
        preprocess_partial,
        batched=True,
        num_proc=4,
        remove_columns=eval_data.column_names
    )

    # Step 3: Define the training arguments
    training_args = TrainingArguments(
        output_dir=script_args.output_dir,

        per_device_train_batch_size=script_args.per_device_batch_size,
        per_device_eval_batch_size=script_args.per_device_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,

        optim="adamw_torch",
        learning_rate=script_args.learning_rate,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.num_warmup_steps,

        fp16=not script_args.no_fp16,
        bf16=script_args.bf16,

        remove_unused_columns=False,
        evaluation_strategy="steps",
        save_strategy="steps" if not script_args.debug else "no",
        max_steps=script_args.max_steps if not script_args.debug else 5,
        eval_steps=script_args.eval_freq if not script_args.debug else 5,
        logging_steps=script_args.logging_steps if not script_args.debug else 1,
        save_steps=script_args.save_freq,
        weight_decay=script_args.weight_decay,

        run_name=script_args.exp_name,
        report_to="wandb" if script_args.log_with == "wandb" else None,
    )

    # Step 4: Define the LoraConfig
    if script_args.use_peft:
        peft_config = LoraConfig(r=script_args.lora_rank, lora_alpha=32, bias="none", task_type="SEQ_CLS", modules_to_save=["scores"])
    else:
        peft_config = None

    # Step 5: Define the Trainer
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=peft_config,
        max_length=script_args.seq_length,
    )

    for batch in trainer.get_train_dataloader():
        break

    example_chosen = tokenizer.batch_decode(batch["input_ids_chosen"])
    example_rejected = tokenizer.batch_decode(batch["input_ids_rejected"])

    for c, r in zip(example_chosen[:2], example_rejected[:2]):
        print_rank_0("====================Example====================")
        print_rank_0("[CHOSEN]:", c)
        print_rank_0("\n")
        print_rank_0("[REJECTED]:", r)
        print_rank_0("====================Example====================\n")

    trainer.train()

if __name__ == "__main__":
    os.environ["WANDB_PROJECT"] = "LaGE"
    main()