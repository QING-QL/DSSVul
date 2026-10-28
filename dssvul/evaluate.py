import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

from .data import make_splits, read_corpus
from .train import build_model, evaluate, seed_all


def evaluate_checkpoint(checkpoint, data, splits, output, device="cpu", model_path=None):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = {**saved["config"], "data": str(data), "device": device}
    if model_path:
        config["model_path"] = str(model_path)
    if not saved.get("split_sha256"):
        raise ValueError("Checkpoint requires the split binding written by this release")
    actual_split_hash = hashlib.sha256(Path(splits).read_bytes()).hexdigest()
    if actual_split_hash != saved["split_sha256"]:
        raise ValueError("Use the unchanged splits.json saved in the checkpoint's run directory")
    seed_all(config["seed"])
    torch.set_num_threads(config["threads"])
    corpus = read_corpus(config)
    if corpus.fingerprint() != saved["split_fingerprint"]:
        raise ValueError("Checkpoint corpus fingerprint mismatch")
    if corpus.metadata["source_sha256"] != saved["metadata"]["source_sha256"]:
        raise ValueError("Checkpoint data SHA256 mismatch")
    partitions = make_splits(corpus, manifest=splits)
    if config["backbone"] == "reveal":
        for key in ("relations", "node_types", "vector_size", "feature_fit_ids", "seed", "encoder_sha256"):
            if corpus.metadata.get(key) != saved["metadata"].get(key):
                raise ValueError("CPG feature metadata mismatch: " + key)
    model = build_model(config, saved["metadata"]).to(device)
    model.load_state_dict(saved["state_dict"], strict=True)
    result, probabilities = evaluate(model, corpus, partitions["test"], config)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "label", "p_vulnerable", "prediction"])
        for index, probability in zip(partitions["test"], probabilities):
            writer.writerow([corpus.ids[index], int(corpus.labels[index]), float(probability[1]), int(probability.argmax())])
    report = {"metrics": result, "method_revision": config.get("method_revision"),
              "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
              "data_sha256": corpus.metadata["source_sha256"], "split_sha256": actual_split_hash}
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved DSSVul model on its fixed test set")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-path")
    args = parser.parse_args()
    print(json.dumps(evaluate_checkpoint(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
