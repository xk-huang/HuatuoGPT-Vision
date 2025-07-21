"""
python merge_json.py outputs/cot/huatuo_vision_34b
"""

import json
from pathlib import Path

import click


@click.command()
@click.argument("output_dir", type=click.Path(exists=True))
def main(output_dir):
    output_dir = Path(output_dir)
    output_shard_dir = output_dir / "shards"

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


if __name__ == "__main__":
    main()
