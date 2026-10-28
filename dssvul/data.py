import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split


class Corpus:
    def __init__(self, inputs, labels, ids, kind, metadata):
        if not np.isin(np.asarray(labels), [0, 1]).all():
            raise ValueError("Labels must be exactly 0 or 1")
        self.inputs, self.labels = inputs, np.asarray(labels, dtype=np.int64)
        self.ids, self.kind, self.metadata = list(map(str, ids)), kind, metadata
        if len(inputs) != len(labels) or len(ids) != len(labels):
            raise ValueError("Input, label, and ID lengths differ")
        if not set(self.labels.tolist()) <= {0, 1}:
            raise ValueError("Only binary labels 0/1 are supported")
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("Duplicate sample IDs: resolve before splitting")

    def batch(self, indices, device):
        if self.kind == "cpg":
            from torch_geometric.data import Batch, Data
            graphs = [Data(x=self.inputs[int(i)].x,
                           edge_index=self.inputs[int(i)].edge_index,
                           edge_type=self.inputs[int(i)].edge_type,
                           num_nodes=len(self.inputs[int(i)].x)) for i in indices]
            return Batch.from_data_list(graphs).to(device)
        if self.kind == "graph":
            from torch_geometric.data import Batch

            from torch_geometric.data import Data
            graphs = [Data(x_type=self.inputs[int(i)].x_type,
                           edge_index=self.inputs[int(i)].edge_index,
                           num_nodes=len(self.inputs[int(i)].x_type)) for i in indices]
            return Batch.from_data_list(graphs).to(device)
        if self.kind == "tokens":
            return {k: v[indices].to(device) for k, v in self.inputs.items()}
        return self.inputs[indices].to(device)

    def fingerprint(self):
        digest = hashlib.sha256()
        for sample_id, target in zip(self.ids, self.labels):
            digest.update(json.dumps([sample_id, int(target)], ensure_ascii=True).encode())
        return digest.hexdigest()


def read_corpus(config):
    path = Path(config["data"])
    backbone = config["backbone"]
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    metadata = {"source": str(path.resolve()), "source_sha256": digest.hexdigest(), "backbone": backbone}
    if backbone == "reveal":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        graphs = payload["graphs"]
        meta = payload["metadata"]
        if not graphs:
            raise ValueError("Empty CPG corpus")
        relations = meta["relations"]
        if not {"AST", "CFG", "REACHING_DEF"} <= set(relations):
            raise ValueError("ReVeal CPG must include AST, CFG, and data-flow relation definitions")
        dim = graphs[0].x.shape[1]
        present = set()
        for g in graphs:
            if g.x.ndim != 2 or len(g.x) == 0 or g.x.shape[1] != dim or not torch.isfinite(g.x).all():
                raise ValueError("Invalid CPG node features")
            if g.edge_index.shape != (2, len(g.edge_type)):
                raise ValueError("Invalid CPG edge shape")
            if g.edge_index.numel() and (g.edge_index.min() < 0 or g.edge_index.max() >= len(g.x)):
                raise ValueError("CPG edge outside node array")
            if g.edge_type.numel() and (g.edge_type.min() < 0 or g.edge_type.max() >= len(relations)):
                raise ValueError("Unknown CPG relation")
            present.update(g.edge_type.tolist())
        if not {relations.index(k) for k in ("AST", "CFG", "REACHING_DEF")} <= present:
            raise ValueError("Corpus claims CPG relations but is missing actual structural/data-flow edges")
        metadata.update(meta)
        metadata.update(input_dim=dim, num_relations=len(relations))
        return Corpus(graphs, [int(g.y.item()) for g in graphs], [g.graph_id for g in graphs], "cpg", metadata)
    if backbone == "ast-ggnn":

        graphs = torch.load(path, map_location="cpu", weights_only=False)
        labels = [int(g.y.item()) for g in graphs]
        ids = [str(getattr(g, "graph_id", i)) for i, g in enumerate(graphs)]
        if any(len(g.x_type) == 0 or int(g.x_type.min()) < 0 for g in graphs):
            raise ValueError("AST node types must be nonempty and nonnegative")
        num_types = max(int(g.x_type.max()) for g in graphs) + 1
        for g in graphs:
            if g.edge_index.ndim != 2 or g.edge_index.shape[0] != 2:
                raise ValueError("edge_index must have shape 2 x E")
            if g.edge_index.numel() and (g.edge_index.min() < 0 or g.edge_index.max() >= len(g.x_type)):
                raise ValueError("AST edge endpoint is outside the node array")
        metadata.update(num_types=num_types, provenance="AST-only, not full ReVeal CPG")
        return Corpus(graphs, labels, ids, "graph", metadata)
    if backbone in ("mlp", "reveal-features"):
        with np.load(path, allow_pickle=False) as data:
            x, y = data["X"], data["y"]
            ids = data["ids"].tolist() if "ids" in data else [str(i) for i in range(len(y))]
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError("X must be a finite N x D feature array")
        metadata.update(input_dim=x.shape[1], provenance="Feature-only; verify upstream encoder training split")
        return Corpus(torch.as_tensor(x, dtype=torch.float32), y, ids, "features", metadata)
    if backbone != "linevul":
        raise ValueError(f"Unknown backbone: {backbone}")
    import pandas as pd
    from transformers import AutoTokenizer
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix in (".json", ".jsonl"):
        frame = pd.read_json(path, lines=path.suffix == ".jsonl")
    else:
        frame = pd.read_csv(path)
    text_col, label_col = config.get("text_column", "functionSource"), config.get("label_column", "label")
    if frame[text_col].isna().any() or frame[label_col].isna().any():
        raise ValueError("Null code/labels must be resolved before training")
    if not frame[text_col].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("Code must contain nonempty strings")
    if not frame[label_col].isin([0, 1]).all():
        raise ValueError("Labels must be exactly 0 or 1")
    frame = frame.copy()
    frame["_content_id"] = frame[text_col].map(lambda s: hashlib.sha256(str(s).encode()).hexdigest())
    if "sample_id" in frame and frame["sample_id"].tolist() != frame["_content_id"].tolist():
        raise ValueError("Sample IDs must match the exact UTF-8 code SHA256")
    label_counts = frame.groupby("_content_id")[label_col].nunique()
    conflicting = label_counts[label_counts > 1].index.tolist()
    metadata["conflicting_content_ids"] = conflicting
    metadata["conflicting_rows_excluded"] = 0
    if conflicting:
        if config.get("conflicting_labels", "error") != "drop":
            raise ValueError(f"Identical code has conflicting labels in {len(conflicting)} groups; "
                             "set conflicting_labels=drop to explicitly exclude all conflicting rows")
        drop_mask = frame["_content_id"].isin(conflicting)
        metadata["conflicting_rows_excluded"] = int(drop_mask.sum())
        frame = frame.loc[~drop_mask].copy()
    before = len(frame)
    if frame["_content_id"].duplicated().any():
        raise ValueError("Resolve exact duplicates explicitly before loading a fixed dataset")
    frame = frame.drop_duplicates("_content_id").reset_index(drop=True)
    metadata["exact_duplicates_removed"] = before - len(frame)

    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], local_files_only=True)
    encoded = tokenizer(frame[text_col].astype(str).tolist(), truncation=True,
                        padding="max_length", max_length=config.get("max_length", 512),
                        return_tensors="pt")

    corpus = object.__new__(Corpus)
    corpus.inputs = dict(encoded)
    corpus.labels = frame[label_col].to_numpy(dtype=np.int64)
    corpus.ids = frame["_content_id"].tolist()
    corpus.kind, corpus.metadata = "tokens", metadata
    if not set(corpus.labels.tolist()) <= {0, 1}:
        raise ValueError("Only binary labels are supported")
    return corpus


def make_splits(corpus, fraction=0.1, seed=42, manifest=None):

    if not 0 < fraction < 1:
        raise ValueError("labeled_fraction must be in (0, 1)")
    if manifest:
        with open(manifest) as stream:
            saved = json.load(stream)
        if saved["fingerprint"] != corpus.fingerprint():
            raise ValueError("Split manifest does not match corpus IDs/labels")
        if any(type(i) is not int for values in saved["indices"].values() for i in values):
            raise ValueError("Split indices must be integers")
        splits = {k: np.asarray(saved["indices"][k], dtype=np.int64)
                  for k in ("labeled", "unlabeled", "validation", "test")}
    else:
        all_idx = np.arange(len(corpus.labels))
        train, heldout = train_test_split(all_idx, test_size=0.2, random_state=seed,
                                         stratify=corpus.labels)
        valid, test = train_test_split(heldout, test_size=0.5, random_state=seed,
                                      stratify=corpus.labels[heldout])
        labeled, unlabeled = train_test_split(train, train_size=fraction, random_state=seed,
                                             stratify=corpus.labels[train])
        splits = dict(labeled=labeled, unlabeled=unlabeled, validation=valid, test=test)
    combined = np.concatenate(list(splits.values()))
    if len(combined) != len(corpus.labels) or set(combined.tolist()) != set(range(len(corpus.labels))):
        raise ValueError("Splits must be disjoint and cover the complete corpus exactly once")
    if manifest:
        for key, indices in splits.items():
            if saved.get("sample_ids", {}).get(key) != [corpus.ids[i] for i in indices]:
                raise ValueError("Split sample IDs/order do not match: " + key)
    for key in ("labeled", "validation", "test"):
        if set(corpus.labels[splits[key]].tolist()) != {0, 1}:
            raise ValueError(f"Both classes required in {key}; increase data/label budget")
    if corpus.kind == "cpg":
        fit_ids = set(corpus.metadata.get("feature_fit_ids", []))
        allowed = {corpus.ids[i] for i in splits["labeled"]}
        if not fit_ids or not fit_ids <= allowed:
            raise ValueError("CPG feature fitting must match labeled training IDs; use the prepared split manifest")
    return splits


def save_splits(path, corpus, splits):
    data = {"fingerprint": corpus.fingerprint(),
            "indices": {k: v.tolist() for k, v in splits.items()},
            "sample_ids": {k: [corpus.ids[i] for i in v] for k, v in splits.items()},
            "sizes": {k: len(v) for k, v in splits.items()},
            "class_counts_for_audit_only": {k: np.bincount(corpus.labels[v], minlength=2).tolist()
                                            for k, v in splits.items()}}
    Path(path).write_text(json.dumps(data, indent=2))
