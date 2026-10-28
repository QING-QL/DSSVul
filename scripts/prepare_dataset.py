import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dssvul.datasets import prepare_dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare function-level data and reproducible partitions")
    parser.add_argument("--dataset", choices=["devign", "reveal", "bigvul", "primevul", "diversevul", "megavul"], required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--source-splits", nargs="+", choices=["train", "validation", "test"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--text-column")
    parser.add_argument("--label-column")
    print(json.dumps(prepare_dataset(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
