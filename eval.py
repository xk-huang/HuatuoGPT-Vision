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
    QUERY_PROMPT = (
        # "{}\n{}\nAnswer with the option's letter from the given choices directly."
        "You will solve a problem/request. You should provide your thoughts within <think> </think> tags before providing the answer.\nWrite your final answer within <answer> </answer> tags.\n{}\n{}"
    )

    def __init__(self, config):
        self.config = config
        max_dataset_size = config.max_dataset_size

        self.dataset = datasets.load_dataset("med-vlrm/med-vlm-eval-v2", split="test")
        if max_dataset_size is not None:
            print(f"max_dataset_size: {max_dataset_size}")
            self.dataset = self.dataset.select(range(max_dataset_size))

    def mmmu_prompt(self, da):
        self.query_prompt = "Please answer the multiple-choice questions below.\n{}\n{}"
        opt_title = ["A", "B", "C", "D", "E", "F", "G"]
        if isinstance(da["options"], str):
            da["options"] = ast.literal_eval(da["options"])
        option_str = "\n".join(
            f"{opt}. {ops}" for opt, ops in zip(opt_title, da["options"])
        )
        return self.query_prompt.format(da["question"], option_str)

    def OmniMedVQA_prompt(self, entity):
        q_str = (
            entity["question"]
            + f'Here are {len(entity["options"])} candidate answers:'
            + str(entity["options"])
            + " Only return what you think is the correct answer from the candidate answers, do not return any other irrelevant text!"
        )
        return q_str

    def llava_prompt(self, da):
        self.query_prompt = (
            "{}\n{}\nAnswer with the option's letter from the given choices directly."
        )
        opt_title = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
        if isinstance(da["options"], str):
            da["options"] = ast.literal_eval(da["options"])
        option_str = "\n".join(
            f"{opt}. {ops}" for opt, ops in zip(opt_title, da["options"])
        )
        return self.query_prompt.format(da["question"], option_str)

    def med_vlrm_prompt(self, row):
        options = row["options"]
        options = json.loads(options)
        question = row["question"]

        option_str = "\n".join(f"{opt}. {ops}" for opt, ops in options.items())
        return self.QUERY_PROMPT.format(question, option_str)

    def __getitem__(self, index):
        da = self.dataset[index]
        query = self.med_vlrm_prompt(da)
        # List of PIL.Image
        image = da["images"]

        return {"data": da, "query": query, "image": image}

    def __len__(self):
        return len(self.dataset)

    def collate_fn(self, batch):
        out_batch = {}
        out_batch["query"] = [x["query"] for x in batch]
        out_batch["data"] = [x["data"] for x in batch]
        out_batch["image"] = [x["image"] for x in batch]
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
        ress = []

        dataloader_iterator = (
            tqdm(val_dataloader, total=len(val_dataloader))
            if accelerator.is_main_process
            else val_dataloader
        )

        for batch in dataloader_iterator:
            for da, query, image in zip(batch["data"], batch["query"], batch["image"]):
                response = bot.inference(
                    query,
                    image,
                )
                da["model_output"] = response[0]
                da.pop("images")
                cache_data.append(da)

        torch.cuda.empty_cache()
        accelerator.wait_for_everyone()
        breakpoint()

        if dist.is_initialized():
            all_data = [None] * dist.get_world_size()
            all_response = [None] * dist.get_world_size()

            dist.all_gather_object(all_data, cache_data)

            all_data = [item for sublist in all_data for item in sublist]
        else:
            all_data = cache_data

        for d in all_data:
            ress.append(d)

        output_dir = args.output_dir
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if accelerator.is_main_process:
            task_name = (
                os.path.basename(args.model_path)
                + f'_{os.path.split(args.data_path)[-1].replace(".json","")}'
            )

            out_file = output_dir / f"{task_name}.json"
            with open(out_file, "w") as fw:
                json.dump(ress, fw, ensure_ascii=False, indent=2)
            print(f"test results: {out_file}")
            print(f"question num: {len(ress)}")
            val_res = score_mix_llava(ress)
            outstr = json.dumps(val_res, ensure_ascii=False, indent=2)
            accelerator.print(outstr)
            out_result_file = output_dir / f"{task_name}_result.json"
            with open(out_result_file, "w") as fw:
                json.dump(val_res, fw, ensure_ascii=False, indent=2)
        else:
            print("Not main process, skip saving results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Args of sft")

    # Model Args
    parser.add_argument(
        "--data_path", default="medical_multimodel_evaluation_data.json", type=str
    )
    parser.add_argument("--model_path", default="HuatuoGPT-Vision-7B", type=str)
    parser.add_argument("--max_new_tokens", default=256, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--max_dataset_size", default=None, type=int)
    parser.add_argument("--output_dir", default="./outputs/huatuo_vision", type=str)

    # Other Args
    parser.add_argument("--seed", default=42, type=int)

    args = parser.parse_args()

    set_seed(args.seed)
    test(args)
