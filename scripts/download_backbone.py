import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description="Download the CodeBERT revision used by the LineVul configuration")
    parser.add_argument("--output", default="pretrained/codebert-base")
    parser.add_argument("--revision", default="3b0952feddeffad0063f274080e3c23d75e7eb39")
    args = parser.parse_args()
    destination = Path(args.output)
    snapshot_download(repo_id="microsoft/codebert-base", revision=args.revision, local_dir=destination,
                      allow_patterns=["config.json", "pytorch_model.bin", "model.safetensors", "vocab.json", "merges.txt", "tokenizer*.json", "special_tokens_map.json"])
    (destination / "source.json").write_text(json.dumps({"repository": "microsoft/codebert-base", "revision": args.revision}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
