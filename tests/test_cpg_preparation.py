import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from dssvul.data import Corpus, make_splits, read_corpus, save_splits
from scripts.prepare_cpg import prepare


class CPGPreparationTests(unittest.TestCase):
    def make_inputs(self, root):
        (root / "src").mkdir()
        sources, graphs = [], []
        for i in range(200):
            code = f"int f{i}() {{\n  return {i};\n}}\n"
            sid = hashlib.sha256(code.encode()).hexdigest()
            filename = sid + ".cpp"
            (root / "src" / filename).write_bytes(code.encode())
            sources.append(dict(id=sid, file=filename, label=i % 2))
            graphs.append(dict(file=filename, nodes=[dict(id="1", type="METHOD", code=f"unique_{i}"),
                                                    dict(id="2", type="RETURN", code="return")],
                               edges=[["1", "2", relation] for relation in ("AST", "CFG", "REACHING_DEF", "CDG")]))
        (root / "sources.json").write_text(json.dumps(sources))
        (root / "graphs.json").write_text(json.dumps(list(reversed(graphs))))
        corpus = Corpus(torch.zeros(200, 1), [r["label"] for r in sources], [r["id"] for r in sources], "features", {})
        split = make_splits(corpus)
        save_splits(root / "splits.json", corpus, split)
        return corpus, split

    def test_fixed_order_split_fitting_and_roundtrip(self):
        from gensim.models import KeyedVectors
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus, split = self.make_inputs(root)
            prepare(root, root / "prepared", root / "splits.json", vector_size=8)
            loaded = read_corpus({"backbone": "reveal", "data": str(root / "prepared/graphs.pt")})
            self.assertEqual(loaded.ids, corpus.ids)
            restored = make_splits(loaded, manifest=root / "prepared/splits.json")
            for key in split:
                self.assertEqual(restored[key].tolist(), split[key].tolist())
            vectors = KeyedVectors.load(str(root / "prepared/node_word_vectors.kv"))
            labeled = set(split["labeled"].tolist())
            for i in range(200):
                self.assertEqual(f"unique_{i}" in vectors, i in labeled)
            self.assertTrue(loaded.metadata["encoder_sha256"])
            self.assertEqual(len(loaded.metadata["relations"]), 8)

    def test_missing_graph_does_not_create_new_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_inputs(root)
            graphs = json.loads((root / "graphs.json").read_text())
            (root / "graphs.json").write_text(json.dumps(graphs[:-1]))
            with self.assertRaisesRegex(ValueError, "no exported graph"):
                prepare(root, root / "prepared", root / "splits.json", vector_size=8)
            self.assertFalse((root / "prepared/graphs.pt").exists())
            self.assertFalse((root / "prepared/splits.json").exists())


if __name__ == "__main__":
    unittest.main()
