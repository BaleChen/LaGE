import argparse
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import AutoPeftModelForCausalLM
from datasets import load_dataset
import torch

from tqdm import tqdm

from functools import partial

def proc_data(examples):
    """Reference: https://github.com/huggingface/trl/pull/444"""

    output_text = []
    for i in range(len(examples["instruction"])):
        instruction = examples["instruction"][i]
        input_text = examples["input"][i]
        response = examples["output"][i]

        if len(input_text) >= 2:
            text = (
                "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
            ).format(instruction=instruction, input=input_text)
        else:
            text = (
                "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:\n"
            ).format(instruction=instruction)
        output_text.append(text)

    return output_text

def tokenize(examples, tokenizer, max_length):
    tokenized_inputs = tokenizer(
        proc_data(examples),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="pt",
    )

    return {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
    }

def main(args):
    # load model, adaptor, and tokenizer
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.lora_path, 
        load_in_8bit=True
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.padding_side = "left"

    # load dataset
    gen_data = load_dataset("json", data_files=args.data_path)["train"]

    tokenize_wrapped = partial(tokenize, tokenizer=tokenizer, max_length=args.seq_length)
    tokenized_dataset = gen_data.map(
        tokenize_wrapped,
        batched=True,
        remove_columns=gen_data.column_names,
        num_proc=4,
        batch_size=50
    )
    # generation
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        do_sample=True,
        top_k=0.0,
        top_p=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    gen_results = []

    for i in tqdm(range(0, len(tokenized_dataset), args.batch_size)):
        if i + args.batch_size > len(tokenized_dataset):
            batch = tokenized_dataset.select(range(i, len(tokenized_dataset)))
        else:
            batch = tokenized_dataset.select(range(i, i + args.batch_size))
        
        input_ids = torch.tensor(batch["input_ids"]).to("cuda")
        attention_mask = torch.tensor(batch["attention_mask"]).to("cuda")
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
        )

        gen_results.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))

    # save results
    out_data = load_dataset("json", data_files=args.data_path)["train"]
    out_data = out_data.add_column("generated", gen_results)
    out_data.to_csv(os.path.join(args.save_path, f"{args.config_name}.csv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--debug", action="store_true", default=False)

    parser.add_argument("--model_name_or_path", type=str, default="yahma/llama-7b-hf")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, default="./out/generation")

    parser.add_argument("--seq_length", type=int, default=768)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--min_new_tokens", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--config_name", type=str, default=None)

    args = parser.parse_args()
    args.config_name = "-".join(args.lora_path.split("/")[-2:])
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    main(args)