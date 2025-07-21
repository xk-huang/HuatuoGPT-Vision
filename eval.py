"""Code for finetune_huatuo"""

import argparse
import ast
import copy
import json
import logging
import os
import random
import shutil
import sys
from pathlib import Path

import datasets
import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator, DeepSpeedPlugin
from cli import HuatuoChatbot
from scorer import score_mix_llava
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelWithLMHead,
    AutoTokenizer,
    set_seed,
)
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList

os.umask(0)


logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")


class TestDataset(torch.utils.data.Dataset):
    QUERY_PROMPT = None

    def __init__(self, config):
        self.config = config
        max_dataset_size = config.max_dataset_size

        self.dataset = datasets.load_dataset("med-vlrm/med-vlm-eval-v2", split="test")
        if max_dataset_size is not None:
            print(f"max_dataset_size: {max_dataset_size}")
            self.dataset = self.dataset.select(range(max_dataset_size))

        if config.use_cot:
            self.QUERY_PROMPT = 'You will solve a problem/request. First think step by step, and then answer with the option letter from the given choices directly after "Answer: ".\n\nQuestions:{}\n\nOptions:{}'
            print(f"Using chain of thought (CoT) prompt: {self.QUERY_PROMPT}")
        else:
            self.QUERY_PROMPT = "Answer with the option's letter from the given choices directly.\n{}\n{}"
            print(f"Using direct answer prompt: {self.QUERY_PROMPT}")

    def med_vlrm_prompt(self, row):
        options = row["options"]
        options = json.loads(options)
        answer_label = row["answer_label"]
        answer = row["answer"]

        # randomly shuffle options
        new_options, new_answer_label = self.shuffle_question(
            options, answer_label, answer
        )
        new_answer = new_options[new_answer_label]
        if new_answer != answer:
            raise ValueError(
                f"Shuffled answer '{new_answer}' does not match original answer '{answer}'."
            )
        question = row["question"]

        option_str = "\n".join(f"{opt}. {ops}" for opt, ops in new_options.items())
        return {
            "data": {
                "question": question,
                "options": new_options,
                "answer_label": new_answer_label,
                "answer": new_answer,
                "dataset_index": row["dataset_index"],
                "dataset_name": row["dataset_name"],
            },
            "query": self.QUERY_PROMPT.format(question, option_str),
            "images": row["images"],
        }

    def shuffle_question(self, options, answer_label, answer):
        """Return a copy of `question_data` with its options shuffled and labels reassigned."""
        # check
        if options[answer_label] != answer:
            raise ValueError(
                f"Answer label '{answer_label}' does not match the answer '{answer}' in options."
            )

        items = list(options.items())  # convert to list of (label, text)
        random.shuffle(items)  # shuffle in-place

        new_labels = sorted([lbl for lbl, _ in items])  # extract shuffled labels
        shuffled = {lbl: text for lbl, (_, text) in zip(new_labels, items)}

        # find which new label corresponds to the original correct answer value
        new_answer_label = next(lbl for lbl, text in shuffled.items() if text == answer)

        return shuffled, new_answer_label

    def __getitem__(self, index):
        da = self.dataset[index]
        return self.med_vlrm_prompt(da)

    def __len__(self):
        return len(self.dataset)

    def collate_fn(self, batch):
        out_batch = {}
        out_batch["query"] = [x["query"] for x in batch]
        out_batch["data"] = [x["data"] for x in batch]
        out_batch["images"] = [x["images"] for x in batch]
        return out_batch


def get_response(inputs, outputs, tokenizer, num_return):
    responses_list = []
    batch_return = []
    for i, output in enumerate(outputs):
        input_len = len(inputs[0])
        generated_output = output[input_len:]
        batch_return.append(
            tokenizer.decode(generated_output, skip_special_tokens=True)
        )
        if i % num_return == num_return - 1:
            responses_list.append(batch_return)
            batch_return = []
    return responses_list


def table_to_csv_string(table):
    rows = [",".join(table.columns)]
    for row in table.data:
        rows.append(",".join(map(str, row)))
    return "\n".join(rows)


def test(args):
    accelerator = Accelerator()

    torch.cuda.set_device(accelerator.process_index)
    accelerator.print(f"args:\n{args}")

    bot = HuatuoChatbot(args.model_path)
    accelerator.print("load_finish")

    bot.gen_kwargs["max_new_tokens"] = args.max_new_tokens

    dataset = TestDataset(args)

    val_dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        collate_fn=dataset.collate_fn,
    )

    val_dataloader = accelerator.prepare(val_dataloader)
    accelerator.wait_for_everyone()
    cache_data = []

    with torch.no_grad():
        dataloader_iterator = (
            tqdm(val_dataloader, total=len(val_dataloader))
            if accelerator.is_main_process
            else val_dataloader
        )

        for batch in dataloader_iterator:
            for da, query, images in zip(
                batch["data"], batch["query"], batch["images"]
            ):
                response = bot.inference(
                    query,
                    images,
                )
                da["model_output"] = response[0]
                cache_data.append(da)

        torch.cuda.empty_cache()

        output_dir = args.output_dir
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if dist.is_initialized():
            rank_id = dist.get_rank()
        else:
            rank_id = 0

        output_shard_dir = output_dir / "shards"
        output_shard_dir.mkdir(parents=True, exist_ok=True)
        output_shard_path = output_shard_dir / f"shard_{rank_id}.json"
        with open(output_shard_path, "w") as fw:
            json.dump(cache_data, fw, ensure_ascii=False, indent=2)
        print(f"shard results: {output_shard_path}")

        if args.no_merge_json:
            print("Skipping merging JSON files as per --no_merge_json flag.")
            return

        # barrier
        dist.barrier()

        if accelerator.is_main_process:
            # merge all shards
            all_data = []
            for shard_file in output_shard_dir.glob("shard_*.json"):
                with open(shard_file, "r") as fr:
                    shard_data = json.load(fr)
                    all_data.extend(shard_data)
            output_path = output_dir / "eval.json"
            with open(output_path, "w") as fw:
                json.dump(all_data, fw, ensure_ascii=False, indent=2)
            print(f"merged results: {output_path}")

        dist.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Args of sft")

    # Model Args
    parser.add_argument(
        "--data_path", default="medical_multimodel_evaluation_data.json", type=str
    )
    parser.add_argument("--model_path", default="HuatuoGPT-Vision-7B", type=str)
    parser.add_argument("--max_new_tokens", default=4096, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--max_dataset_size", default=None, type=int)
    parser.add_argument("--output_dir", default="./outputs/huatuo_vision", type=str)
    parser.add_argument("--use_cot", action="store_true", help="Use chain of thought")

    # Other Args
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--no_merge_json", action="store_true", help="Do not merge json files"
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    set_seed(args.seed)
    test(args)
