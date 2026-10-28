import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from dssvul.core import nrce_loss
from dssvul.data import Corpus, make_splits, save_splits
from dssvul.datasets import prepare_dataset, records, normalize, validate_dataset
from dssvul.evaluate import evaluate_checkpoint
from dssvul.models import FeatureMLP
from dssvul.train import DEFAULTS, fit, infer, run, seed_all


class ReleaseTests(unittest.TestCase):
    def test_nce_uses_unscaled_unit_embeddings(self):
        corpus = Corpus(torch.randn(20, 3), [0, 1] * 10, range(20), "features", {})
        model = FeatureMLP(3, 4)
        z, _, weights = infer(model, corpus, np.arange(20), DEFAULTS, uncertainty=False)
        self.assertTrue(torch.equal(weights, torch.zeros(20)))
        self.assertTrue(torch.equal((1 + weights[:, None]) * z, z))

    def test_nbl_update_equals_nrce_without_student_focal(self):
        seed_all(81)
        corpus = Corpus(torch.randn(12, 3), [0, 1] * 6, range(12), "features", {})
        model = FeatureMLP(3, 4)
        expected = copy.deepcopy(model)
        config = {**DEFAULTS, "variant": "nbl", "student_epochs": 1, "batch_size": 8,
                  "micro_batch_size": 8, "seed": 81}
        order = np.random.default_rng(82).permutation(8)
        optimizer = torch.optim.AdamW(expected.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        nrce_loss(expected(corpus.inputs[order]), torch.tensor(corpus.labels[order])).backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as root:
            pseudo = np.array([False] * 4 + [True] * 4)
            fit(model, corpus, np.arange(8), corpus.labels[:8], pseudo, pseudo,
                torch.tensor([0.8, 0.2]), np.arange(8, 12), config, "student_1", Path(root), torch.tensor([0.1, 0.9]))
        for actual, reference in zip(model.parameters(), expected.parameters()):
            self.assertTrue(torch.allclose(actual, reference, atol=1e-7))

    def test_checkpoint_evaluation_and_wrong_split_rejection(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            data = root / "data.npz"
            np.savez(data, X=np.random.default_rng(91).normal(size=(200, 3)).astype(np.float32), y=[0, 1] * 100)
            config = {**DEFAULTS, "backbone": "mlp", "variant": "ndp", "hidden": 4,
                      "teacher_epochs": 1, "student_epochs": 1, "max_rounds": 1, "threads": 1,
                      "data": str(data), "output": str(root / "train")}
            trained = run(config)
            with patch("dssvul.train.fit", side_effect=AssertionError("Evaluation must not train")):
                evaluated = evaluate_checkpoint(root / "train/best.pt", data, root / "train/splits.json", root / "eval")
            self.assertEqual(evaluated["metrics"], trained["test"])
            self.assertEqual((root / "eval/predictions.csv").read_bytes(), (root / "train/test_predictions.csv").read_bytes())
            wrong = root / "wrong.json"
            saved = json.loads((root / "train/splits.json").read_text())
            saved["indices"]["validation"], saved["indices"]["test"] = saved["indices"]["test"], saved["indices"]["validation"]
            wrong.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, "unchanged splits"):
                evaluate_checkpoint(root / "train/best.pt", data, wrong, root / "rejected")

    def test_dataset_roundtrip_uses_fixed_ids_and_raw_megavul_labels(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            raw = root / "raw.json"
            rows = [dict(func=f"int f{i}() {{ return {i}; }}", func_before=f"int v{i}() {{ return {i}; }}",
                         is_vul=bool(i % 2), repo_name="p", commit_hash=str(i)) for i in range(200)]
            raw.write_text(json.dumps(rows), encoding="utf-8")
            result = prepare_dataset("megavul", [raw], root / "prepared")
            self.assertEqual(result["rows"], 200)
            self.assertEqual(result["cross_partition_groups"], 0)
            self.assertEqual(normalize(rows[1], "megavul")["code"], rows[1]["func_before"])
            canonical = root / "prepared/canonical.csv"
            frozen = root / "prepared/splits_seed42.json"
            restored = prepare_dataset("megavul", [canonical], root / "restored", splits=frozen)
            self.assertEqual(restored["fingerprint"], result["fingerprint"])
            self.assertEqual(restored["split_counts_NV_V"], result["split_counts_NV_V"])

    def test_primevul_retains_source_partitions(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            raw = root / "raw.jsonl"
            rows = [dict(func=f"int f{i}() {{return {i};}}", target=i % 2,
                         split="train" if i < 160 else "valid" if i < 180 else "test") for i in range(200)]
            raw.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            report = prepare_dataset("primevul", [raw], root / "prepared")
            self.assertEqual(report["source_partition_differences"], 0)
            self.assertEqual(report["split_counts_NV_V"]["test"], [10, 10])
            with self.assertRaises(ValueError):
                normalize(dict(func="f", target=2), "primevul")

    def test_conflicting_labels_and_blank_rows_are_explicit(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            rows = [dict(func=f"int f{i}() {{return {i};}}", target=i % 2) for i in range(200)]
            rows.extend([dict(func=" ", target=0), dict(func="int conflict;", target=0), dict(func="int conflict;", target=1), rows[0]])
            raw = root / "raw.jsonl"
            raw.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            report = prepare_dataset("devign", [raw], root / "prepared")
            self.assertEqual(report["rows"], 200)
            self.assertEqual(report["processing"]["blank_rows"], 1)
            self.assertEqual(report["processing"]["conflicting_groups"], 1)
            self.assertEqual(report["processing"]["duplicate_rows"], 1)

    def test_split_id_tampering_rejected(self):
        corpus = Corpus(torch.randn(200, 3), [0, 1] * 100, range(200), "features", {})
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "splits.json"
            save_splits(path, corpus, make_splits(corpus))
            saved = json.loads(path.read_text())
            saved["sample_ids"]["test"][0] = "changed"
            path.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, "sample IDs/order"):
                make_splits(corpus, manifest=path)


if __name__ == "__main__":
    unittest.main()
