from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.distributed as dist
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, AutoPeftModelForSequenceClassification
from tqdm import tqdm
from transformers import Adafactor, AutoTokenizer, HfArgumentParser, pipeline, AutoModelForSequenceClassification

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler

import pdb
import os

tqdm.pandas()

def print_rank_0(*args, **kwargs) -> None:
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)

@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(default="yahma/llama-7b-hf", metadata={"help": "the model name"})
    tokenizer_name: Optional[str] = field(default="yahma/llama-7b-hf", metadata={"help": "the tokenizer name"})
    reward_model_path: Optional[str] = field(default="", metadata={"help": "the reward model path"})

    data_path: Optional[str] = field(default="/scratch/bc3088/LF-research/LaGE/data/ILF-refinement-sft", metadata={"help": "the data path"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    lora_rank: Optional[int] = field(default=8, metadata={"help": "the rank of the LoRA matrix"})
    learning_rate: Optional[float] = field(default=1e-4, metadata={"help": "the learning rate"})
    output_max_length: Optional[int] = field(default=128, metadata={"help": "maximum length for generation"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"}) 
    batch_size: Optional[int] = field(default=8, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(default=4, metadata={"help": "the number of ppo epochs"})
    gradient_accumulation_steps: Optional[int] = field(
        default=8, metadata={"help": "the number of gradient accumulation steps"}
    )
    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    batched_gen: Optional[bool] = field(default=False, metadata={"help": "whether to use the batched text gen"})
    save_freq: Optional[int] = field(default=500, metadata={"help": "n steps to save the model"})
    output_dir: Optional[str] = field(default="runs/", metadata={"help": "n steps to save the model"})
    seed: Optional[int] = field(default=42, metadata={"help": "the seed"})
    steps: Optional[int] = field(default=5000, metadata={"help": "number of epochs"})
    init_kl_coef: Optional[float] = field(
        default=0.0,
        metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    )

    adap_kl_ctrl: Optional[bool] = field(default=True, metadata={"help": "Use adaptive KL control, otherwise linear"})


parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]


config = PPOConfig(
    steps=script_args.steps,
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    batch_size=script_args.batch_size,
    mini_batch_size=script_args.mini_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optimize_cuda_cache=True,
    early_stopping=script_args.early_stopping,
    target_kl=script_args.target_kl,
    ppo_epochs=script_args.ppo_epochs,
    seed=script_args.seed,
    init_kl_coef=script_args.init_kl_coef,
    adap_kl_ctrl=script_args.adap_kl_ctrl,
)

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 16,
    "truncation": True,
    # "padding": "max_length",
    "max_length": 768+256, 
}

tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name, add_eos_token=True)

if getattr(tokenizer, "pad_token", None) is None:
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.padding_side = "left"


# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(
    tokenizer,
    data_path=None,
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        data_path (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    if data_path == "Anthropic/hh-rlhf":
        ds = load_dataset(data_path, data_dir="helpful-base", split="train") # Using only a subset
    else:
        ds = load_dataset("json", data_files={"train": os.path.join(data_path, "train.jsonl")})["train"]
    
    original_columns = ds.column_names
    num_proc = 12

    def preprocess_function_instruction(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for i in range(len(examples["instruction"])):
            instruction = examples["instruction"][i]
            input_text = examples["input"][i]
            query = (
                "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
            ).format(instruction=instruction, input=input_text)

            tokenized_question = tokenizer(query, truncation=True, max_length=768)
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_question["input_ids"][:-1])

        return new_examples

    def preprocess_function_hh(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for i in range(len(examples["chosen"])):
            chosen_text = examples["chosen"][i]
            query = "\n\nAssistant:".join(chosen_text.split("\n\nAssistant:")[:-1]) + "\n\nAssistant:"
            input_ids = tokenizer(query, truncation=True, max_length=768)["input_ids"][:-1]
            new_examples["query"].append(query)
            new_examples["input_ids"].append(input_ids)
        return new_examples

    if data_path == "Anthropic/hh-rlhf":
        ds = ds.map(
            preprocess_function_hh,
            batched=True,
            num_proc=num_proc,
            remove_columns=original_columns,
        )
    else:
        ds = ds.map(
            preprocess_function_instruction,
            batched=True,
            num_proc=num_proc,
            remove_columns=original_columns,
        )
    # ds = ds.filter(lambda x: len(x["input_ids"]) < 768, batched=False)

    ds.set_format(type="torch")
    return ds

# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(tokenizer, data_path=script_args.data_path)

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


# set seed before initializing value head for deterministic eval
set_seed(config.seed)

# Now let's build the model, the reference model, and the tokenizer.
current_device = Accelerator().local_process_index

lora_config = LoraConfig(
    r=script_args.lora_rank,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    load_in_8bit=True,
    device_map={"": current_device},
    peft_config=lora_config,
)

model.v_head = model.v_head.to(current_device) # This is a remedy for a bug in the trl code.

optimizer = None
if script_args.adafactor:
    optimizer = Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=config.learning_rate,
    )
# We then build the PPOTrainer, passing the model, the reference model, the tokenizer

ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer,
)

# We then build the sentiment analysis pipeline using our reward model, passing the
# model name and the sentiment analysis pipeline arguments. Let's also make sure to
# set the device to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a ` pipeline` bug

reward_model = AutoModelForSequenceClassification.from_pretrained(
    script_args.reward_model_path,
    load_in_8bit=True,
    trust_remote_code=True,
    num_labels=1,
)
reward_model.eval()

sentiment_pipe = pipeline(
    "text-classification",
    model=reward_model,
    device_map={"": current_device},
    model_kwargs={"load_in_8bit": True},
    tokenizer=tokenizer,
    return_token_type_ids=False,
)

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id
}
output_min_length = 32
output_max_length = script_args.output_max_length
output_length_sampler = LengthSampler(output_min_length, output_max_length)

steps_per_epoch = len(ppo_trainer.dataloader)
print("INFO: Steps per epoch = ", steps_per_epoch)

for epoch in range(script_args.steps // steps_per_epoch):
    print(f"==Epoch {epoch}==")

    for t, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        question_tensors = batch["input_ids"]

        response_tensors = ppo_trainer.generate(
            question_tensors,
            return_prompt=False,
            length_sampler=output_length_sampler,
            **generation_kwargs,
        )
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True) # TODO Test if turning it false make a diff
        
        # Compute reward score (using the sentiment analysis pipeline)
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        rewards = [torch.tensor(output[0]["score"] - script_args.reward_baseline) for output in pipe_outputs]
        # Run PPO step
        stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

        global_step = epoch * steps_per_epoch + t
        if script_args.save_freq and global_step % script_args.save_freq == 0:
            ppo_trainer.save_pretrained(script_args.output_dir + f"step_{epoch}")