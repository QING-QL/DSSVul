import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .train import build_model


@torch.no_grad()
def predict(checkpoint, data, device="cpu", model_path=None):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = dict(saved["config"])
    if model_path:
        config["model_path"] = str(model_path)
    model = build_model(config, saved["metadata"]).to(device).eval()
    model.load_state_dict(saved["state_dict"])
    kind = config["backbone"]
    if kind == "linevul":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config["model_path"], local_files_only=True)
        code = Path(data).read_text(encoding="utf-8")
        x = tokenizer(code, truncation=True, padding="max_length", max_length=config.get("max_length", 512), return_tensors="pt")
        probabilities = model({k: v.to(device) for k, v in x.items()}).softmax(-1).cpu()
        ids = [str(data)]
    else:
        if kind in ("ast-ggnn", "reveal"):
            from torch_geometric.data import Batch, Data
            graphs = torch.load(data, map_location="cpu", weights_only=False)
            if kind == "reveal":
                for key in ("relations", "node_types", "vector_size", "feature_fit_ids", "seed", "encoder_sha256"):
                    if graphs["metadata"].get(key) != saved["metadata"].get(key):
                        raise ValueError("CPG feature encoder mismatch: " + key)
                graphs = graphs["graphs"]
            ids = [str(getattr(g, "graph_id", i)) for i, g in enumerate(graphs)]
            if kind == "reveal":
                inputs = [Data(x=g.x, edge_index=g.edge_index, edge_type=g.edge_type, num_nodes=len(g.x)) for g in graphs]
            else:
                inputs = [Data(x_type=g.x_type, edge_index=g.edge_index, num_nodes=len(g.x_type)) for g in graphs]
        else:
            with np.load(data, allow_pickle=False) as array:
                inputs = torch.tensor(array["X"], dtype=torch.float32)
                ids = array["ids"].astype(str).tolist() if "ids" in array else list(map(str, range(len(inputs))))
        parts = []
        for i in range(0, len(inputs), config["inference_batch_size"]):
            x = inputs[i:i + config["inference_batch_size"]]
            if kind in ("ast-ggnn", "reveal"):
                x = Batch.from_data_list(x)
            parts.append(model(x.to(device)).softmax(-1).cpu())
        probabilities = torch.cat(parts)
    return [{"sample_id": str(i), "p_vulnerable": float(p[1]), "prediction": int(p.argmax())}
            for i, p in zip(ids, probabilities)]


def main():
    parser = argparse.ArgumentParser(description="Predict from a DSSVul checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True, help="LineVul: source file; AST: PyG .pt; feature models: NPZ with X")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-path")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = predict(args.checkpoint, args.data, args.device, args.model_path)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)


if __name__ == "__main__":
    main()
