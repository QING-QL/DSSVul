import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from gensim.models import Word2Vec
from torch_geometric.data import Data
from dssvul.data import Corpus, make_splits, save_splits
from dssvul.datasets import read_manifest, manifest_order, sha256


RELATIONS = ["AST", "CFG", "REACHING_DEF", "CDG"]


def tokens(code):
    return re.findall(r"[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?|[^\w\s]", code)


def prepare(root, output, splits, seed=42, labeled_fraction=0.1, vector_size=64):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((root / "sources.json").read_text(encoding="utf-8"))
    sources = {r["file"]: r for r in manifest}
    for r in manifest:
        if hashlib.sha256((root / "src" / r["file"]).read_bytes()).hexdigest() != r["id"]:
            raise ValueError("Source file SHA256 mismatch")
    raw = json.loads((root / "graphs.json").read_text(encoding="utf-8"))
    graphs_by_file = {}
    for g in raw:
        name = Path(g["file"]).name
        if name in sources and g["nodes"]:
            if name in graphs_by_file:
                raise ValueError("Duplicate exported file: " + name)
            graphs_by_file[name] = g
    saved = read_manifest(splits)
    fixed_ids = manifest_order(saved)
    file_by_id = {r["id"]: r["file"] for r in manifest}
    if len(file_by_id) != len(manifest) or set(fixed_ids) != set(file_by_id):
        raise ValueError("CPG sources must match the complete fixed corpus")
    missing = sorted(set(sources) - set(graphs_by_file))
    if missing:
        (output / "missing_graphs.json").write_text(json.dumps(missing, indent=2), encoding="utf-8")
        raise ValueError("Some fixed samples have no exported graph; inspect missing_graphs.json")
    names = [file_by_id[sid] for sid in fixed_ids]
    y = [sources[n]["label"] for n in names]
    ids = [sources[n]["id"] for n in names]
    dummy = Corpus(torch.zeros(len(names), 1), y, ids, "features", {})
    (output / "input_splits.json").write_text(json.dumps(saved), encoding="utf-8")
    split = make_splits(dummy, labeled_fraction, seed, output / "input_splits.json")
    fit_names = [names[i] for i in split["labeled"]]
    sentences = [tokens(node["code"]) for name in fit_names for node in graphs_by_file[name]["nodes"]]
    sentences = [s for s in sentences if s]
    if not sentences:
        raise ValueError("No labeled-training code tokens")

    w2v = Word2Vec(sentences, vector_size=vector_size, window=5, min_count=1,
                   sg=1, workers=1, seed=seed, epochs=10,
                   hashfxn=lambda s: int(hashlib.sha256(s.encode()).hexdigest()[:8], 16))
    types = ["<UNK>"] + sorted({node["type"] for name in fit_names for node in graphs_by_file[name]["nodes"]})
    vocabulary = {t: i for i, t in enumerate(types)}
    relations = RELATIONS + ["REV_" + r for r in RELATIONS]
    edge_counts, unknown_tokens, all_tokens = Counter(), 0, 0
    result = []
    for name in names:
        g = graphs_by_file[name]
        index = {str(n["id"]): i for i, n in enumerate(g["nodes"])}
        if len(index) != len(g["nodes"]):
            raise ValueError("Duplicate node ID")
        x = np.zeros((len(index), vector_size + len(types)), dtype=np.float32)
        for i, n in enumerate(g["nodes"]):
            ts = tokens(n["code"])
            vectors = [w2v.wv[t] for t in ts if t in w2v.wv]
            all_tokens += len(ts); unknown_tokens += len(ts) - len(vectors)
            if vectors:
                x[i, :vector_size] = np.mean(vectors, axis=0)
            x[i, vector_size + vocabulary.get(n["type"], 0)] = 1
        edges, kinds = [], []
        for source, target, relation in g["edges"]:
            if relation not in RELATIONS:
                continue

            u, v = index[str(source)], index[str(target)]
            k = RELATIONS.index(relation)
            edges.extend([[u, v], [v, u]])
            kinds.extend([k, k + len(RELATIONS)])
            edge_counts[relation] += 1
        edge_index = torch.tensor(edges, dtype=torch.long).reshape(-1, 2).T.contiguous()
        result.append(Data(x=torch.from_numpy(x), edge_index=edge_index,
                           edge_type=torch.tensor(kinds, dtype=torch.long),
                           y=torch.tensor([sources[name]["label"]]), graph_id=sources[name]["id"], num_nodes=len(index)))
    if any(edge_counts[r] == 0 for r in ("AST", "CFG", "REACHING_DEF")):
        raise ValueError("Export lacks actual CPG relations: " + str(edge_counts))
    meta = dict(format="dssvul-cpg-v1", relations=relations, vector_size=vector_size, node_types=types,
                feature_fit_ids=[ids[i] for i in split["labeled"]], seed=seed,
                labeled_fraction=labeled_fraction, feature_fitting="Word2Vec and type vocabulary fitted on D_L only",
                edge_counts=dict(edge_counts), input_files=len(sources), exported_graphs=len(result),
                missing_source_files=sorted(set(sources) - set(names)),
                oov_fraction=unknown_tokens / max(all_tokens, 1),
                provenance="Joern real CPG, AST/CFG/REACHING_DEF/CDG and explicit reverse relations",
                exporter_sha256=hashlib.sha256((root / "graphs.json").read_bytes()).hexdigest())
    w2v.wv.save(str(output / "node_word_vectors.kv"))
    meta["encoder_sha256"] = {p.name: sha256(p) for p in sorted(output.glob("node_word_vectors.kv*"))}
    torch.save(dict(graphs=result, metadata=meta), output / "graphs.pt")
    save_splits(output / "splits.json", dummy, split)
    (output / "manifest.json").write_text(json.dumps(meta, indent=2))

    import pandas as pd
    pd.DataFrame([dict(functionSource=(root / "src" / n).read_bytes().decode("utf-8"), label=sources[n]["label"])
                  for n in names]).to_parquet(output / "code.parquet", index=False)
    print(json.dumps({k: meta[k] for k in ("exported_graphs", "edge_counts", "missing_source_files", "oov_fraction")}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--labeled-fraction", type=float, default=0.1)
    p.add_argument("--vector-size", type=int, default=64)
    a = p.parse_args()
    prepare(a.root, a.output, a.splits, a.seed, a.labeled_fraction, a.vector_size)
