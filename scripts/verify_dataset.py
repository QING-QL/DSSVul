import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dssvul.datasets import validate_dataset


def main():
    parser = argparse.ArgumentParser(description="Verify fixed IDs, labels, partitions and group membership")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--splits", required=True)
    print(json.dumps(validate_dataset(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
