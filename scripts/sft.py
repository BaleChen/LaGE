import argparse
import os
import json
from time import gmtime, strftime
from functools import partial

from accelerate import Accelerator
from accelerate.logging import get_logger

import torch.distributed as dist
from datasets import load_dataset
from peft import LoraConfig
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, logging, set_seed, TrainerCallback

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import pdb


def print_rank_0(*args, **kwargs) -> None:
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)

def prepare_exp_name(args):
    exp_name = "sft"
    exp_name += f"-lr{args.learning_rate}"
    exp_name += f"-bs{args.per_device_batch_size}"
    exp_name += f"-nsteps{args.max_steps}"
    exp_name += f"-r{args.lora_rank}"
    return exp_name

def formatting_prompts_func(examples):
    """Reference: https://github.com/huggingface/trl/pull/444"""

    output_text = []
    for i in range(len(examples["instruction"])):
        instruction = examples["instruction"][i]
        input_text = examples["input"][i]
        response = examples["output"][i]

        if len(input_text) >= 2:
            text = (
                "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
            ).format(instruction=instruction, input=input_text, output=response)
        else:
            text = (
                "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:\n{output}"
            ).format(instruction=instruction, output=response)
        output_text.append(text)

    return output_text

def formatting_dialogue_func(examples, tokenizer, seq_length):
    out = []
    for i in range(len(examples["chosen"])):
        if len(tokenizer.encode(examples["chosen"][i])) <= seq_length:
            out.append(examples["chosen"][i])
    return out
    
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print_rank_0(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            control.should_evaluate = True

def main(args):
    logger = get_logger(__file__, log_level="DEBUG" if args.debug else "INFO")

    # model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        load_in_8bit=True, 
        device_map={"":Accelerator().process_index}
    )

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, add_eos_token=True)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.padding_side = "right"
    
    # dataset
    if args.data_path == "Anthropic/hh-rlhf":
        data = load_dataset(args.data_path, data_dir="helpful-base") # Using only a subset
        data["validation"] = data["test"]
        del data["test"]
    else:
        jsonl_files = {
            "train": os.path.join(args.data_path, "train.jsonl"),
            "validation": os.path.join(args.data_path, "validation.jsonl"),
        }
        data = load_dataset("json", data_files=jsonl_files)

    train_set_size = int(len(data["train"]) * args.train_pct) if not args.debug else 50
    train_data = data["train"].select(range(train_set_size))
    val_data = data["validation"].select(range(25)) if args.debug else data["validation"]

    if args.data_format == "dialogue":
        response_template_ids = tokenizer.encode("\n\nAssistant:", add_special_tokens=False)[1:] # Wordaround for 29871
    elif args.data_format == "instruction":
        response_template_ids = tokenizer.encode("\n\n### Response:\n", add_special_tokens=False)[3:] 
    # NOTE: The first token id is for some reason a empty string 29871, which we need to remove
    # Otherwise, the data collator can't find this response template in the input_ids. This is a bug
    # and I will file a issue to trl repo later. The above code is just a temporary work around.

    # trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        evaluation_strategy="steps",
        max_steps=args.max_steps,
        eval_steps=args.eval_freq if not args.debug else 5,
        save_steps=args.save_freq,
        save_strategy="steps" if not args.debug else "no",
        logging_steps=args.log_freq,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        fp16=not args.no_fp16,
        bf16=args.bf16,
        weight_decay=args.weight_decay,
        
        run_name=args.exp_name,
        report_to="wandb",
        ddp_find_unused_parameters=False,
    )

    partial_formatting_dialogue_func = partial(formatting_dialogue_func, tokenizer=tokenizer, seq_length=args.seq_length)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=lora_config,
        formatting_func=formatting_prompts_func if args.data_format == "instruction" else partial_formatting_dialogue_func,
        max_seq_length=args.seq_length, 
        data_collator=DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer, pad_to_multiple_of=args.seq_length),
        packing=False,
    )

    print_trainable_parameters(trainer.model)

    trainer.add_callback(EvaluateFirstStepCallback())

    for batch in trainer.get_train_dataloader():
        break
    # print_rank_0(batch["input_ids"])
    # print_rank_0(batch["labels"])
    example_prompts = tokenizer.batch_decode(batch["input_ids"])
    for p, l in zip(example_prompts[:2], batch["labels"][:2]):
        print_rank_0("====================Example====================")
        print_rank_0("[PROMPT]:", p)
        print_rank_0("\n")
        print_rank_0("[LABEL]:", tokenizer.decode(l[l != -100]))
        print_rank_0("====================Example====================\n")

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--exp_name", default=None)

    parser.add_argument("--model_name_or_path", type=str, default="yahma/llama-7b-hf")
    parser.add_argument("--data_path", type=str, default="./data/ILF-refinement-sft")
    parser.add_argument("--data_format", type=str, default="instruction")
    parser.add_argument("--train_pct", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default="./out")
    parser.add_argument("--lora_rank", type=int, default=8)

    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--eval_freq", type=int, default=50)
    parser.add_argument("--save_freq", type=int, default=100)

    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--no_fp16", action="store_false")
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--no_gradient_checkpointing", action="store_false", default=False)
    parser.add_argument("--seed", type=int, default=0)
    
    args = parser.parse_args()

    if not args.debug:
        date_time = strftime("%Y-%m-%d-%H:%M", gmtime())
        args.exp_name = prepare_exp_name(args)
        args.output_dir = os.path.join(args.output_dir, "sft", date_time+"-"+args.exp_name)
        try:
            os.makedirs(args.output_dir)
        except FileExistsError:
            pass
        
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(args.__dict__, f, indent=4)
    else:
        args.exp_name = "debug-session"
        args.max_steps = 10

    os.environ["WANDB_PROJECT"] = "LaGE"

    main(args)