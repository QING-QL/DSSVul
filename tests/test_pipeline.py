import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from dssvul.train import run, DEFAULTS, METHOD_REVISION, build_model, evaluate
from dssvul.data import read_corpus, make_splits
from dssvul.core import Selection


class PipelineTests(unittest.TestCase):
    def exercise(self, promote, empty=False):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            data = root / "data.npz"
            np.savez(data, X=np.random.default_rng(1).normal(size=(200, 3)).astype(np.float32), y=[0, 1] * 100)
            output = root / "run"
            stages, evaluations = [], []
            def fake_fit(model, corpus, ids, targets, pseudo, eligible, prior, validation, config, stage, out, alpha):
                stages.append(stage)

                with torch.no_grad():
                    next(model.parameters()).fill_(1 if stage == "teacher" else 2)
                return {"mcc": 0.2 if stage == "teacher" else (0.3 if promote else 0.1)}
            def fake_eval(model, corpus, ids, config):
                evaluations.append(ids.tolist())
                return {"mcc": 0.0}, torch.tensor([[0.6, 0.4]] * len(ids))
            config = dict(backbone="mlp", data=str(data), output=str(output), max_rounds=1,
                          variant="ndp", threads=1, hidden=4)
            with patch("dssvul.train.fit", side_effect=fake_fit), patch("dssvul.train.evaluate", side_effect=fake_eval):
                if empty:
                    def empty_selection(z, p, w, zl, yl, *a):
                        n = len(z); mask = torch.zeros(n, dtype=torch.bool)
                        return Selection(p.argmax(1), 1-p.argmax(1), mask, w, mask, torch.tensor([0.5, 0.5]), 0.8, 3)
                    with patch("dssvul.train.align_labels", side_effect=empty_selection):
                        summary = run(config)
                else:
                    summary = run(config)
            split = json.loads((output / "splits.json").read_text())
            self.assertEqual(evaluations, [split["indices"]["test"]])
            saved = torch.load(output / "best.pt", weights_only=False)
            expected = 2 if promote else 1
            self.assertTrue((next(iter(saved["state_dict"].values())) == expected).all())
            self.assertEqual(stages, ["teacher", "student_1"])
            self.assertEqual(summary["rounds"][0]["promoted"], promote)
            self.assertEqual(summary["rounds"][0]["empty_selection"], empty)
            with self.assertRaises(FileExistsError):
                run(config)

    def test_promote_only_improved_student(self):
        self.exercise(True)

    def test_retain_teacher_on_regression(self):
        self.exercise(False)

    def test_empty_selection_still_trains_student_and_keeps_better_teacher(self):
        self.exercise(False, empty=True)

    def test_rejection_tie_and_empty_selection_do_not_end_rounds(self):

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            data = root / "data.npz"
            np.savez(data, X=np.ones((200, 3), dtype=np.float32), y=[0, 1] * 100)
            stages, inputs, inference_seeds = [], [], []


            scores = {"teacher": 0.2, "student_1": 0.1, "student_2": 0.2,
                      "student_3": 0.21, "student_4": 0.205}

            def fake_fit(model, corpus, ids, targets, pseudo, eligible, prior, valid, config, stage, out, alpha):
                stages.append(stage)
                inputs.append((stage, float(next(model.parameters()).flatten()[0]), int(pseudo.sum())))
                if stage == "student_2":
                    self.assertFalse(pseudo.any())
                    self.assertFalse(eligible.any())
                    self.assertTrue(np.array_equal(targets, corpus.labels[ids]))
                with torch.no_grad():
                    next(model.parameters()).fill_(1 if stage == "teacher" else 1 + int(stage.split("_")[-1]))
                return {"mcc": scores[stage]}

            def fake_infer(model, corpus, ids, config, uncertainty=False, noise_seed=None):
                if noise_seed is not None:
                    inference_seeds.append(noise_seed)
                n = len(ids)
                return torch.ones(n, 4), torch.tensor([[0.6, 0.4]] * n), torch.ones(n)

            def fake_selection(z, probabilities, weights, zl, yl, clusters, seed, use_alignment):
                n = len(z)
                mask = torch.full((n,), seed != 44, dtype=torch.bool)
                return Selection(probabilities.argmax(1), probabilities.argmax(1), mask,
                                 weights, mask, torch.tensor([0.5, 0.5]), 0.6, clusters)

            config = dict(backbone="mlp", data=str(data), output=str(root / "run"),
                          max_rounds=4, min_delta=0.1, hidden=4, threads=1)
            with patch("dssvul.train.fit", side_effect=fake_fit), \
                 patch("dssvul.train.infer", side_effect=fake_infer), \
                 patch("dssvul.train.align_labels", side_effect=fake_selection), \
                 patch("dssvul.train.evaluate", return_value=({"mcc": 0.0}, torch.tensor([[0.6, 0.4]] * 20))) as test_eval:
                result = run(config)
            self.assertEqual(stages, ["teacher", "student_1", "student_2", "student_3", "student_4"])
            self.assertEqual([row[1] for row in inputs[1:]], [1., 1., 1., 4.])
            self.assertEqual([r["promoted"] for r in result["rounds"]], [False, False, True, False])
            self.assertEqual(inference_seeds, [43, 44, 45, 46])
            self.assertEqual(result["stop_reason"], "maximum_rounds")
            self.assertEqual(result["validation"]["mcc"], 0.21)
            self.assertEqual(result["config"]["method_revision"], METHOD_REVISION)
            self.assertEqual(len(list((root / "run").glob("selection_*.npz"))), 4)
            test_eval.assert_called_once()
            split = json.loads((root / "run" / "splits.json").read_text())
            self.assertEqual(test_eval.call_args.args[2].tolist(), split["indices"]["test"])
            saved = torch.load(root / "run" / "best.pt", weights_only=False)
            self.assertTrue((next(iter(saved["state_dict"].values())) == 4).all())

    def test_all_ablation_paths_and_reload(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            data = root / "data.npz"
            rng = np.random.default_rng(4)
            y = np.array([0, 1] * 100)
            x = rng.normal(size=(200, 3)).astype(np.float32) + y[:, None] * 2
            np.savez(data, X=x.astype(np.float32), y=y)
            for variant in ("full", "nce", "ndp", "nbl", "nnrce"):
                with self.subTest(variant=variant):
                    config = {**DEFAULTS, "data": str(data), "output": str(root / variant),
                              "backbone": "mlp", "variant": variant, "teacher_epochs": 1,
                              "student_epochs": 1, "max_rounds": 1, "hidden": 8,
                              "batch_size": 32, "micro_batch_size": 7, "threads": 1}
                    result = run(config)
                    self.assertEqual(result["status"], "completed")
                    saved = torch.load(root / variant / "best.pt", weights_only=False)
                    restored = build_model(saved["config"], saved["metadata"])
                    restored.load_state_dict(saved["state_dict"])
                    corpus = read_corpus(config)
                    split = make_splits(corpus, manifest=root / variant / "splits.json")
                    measured, _ = evaluate(restored, corpus, split["test"], config)
                    self.assertEqual(measured, result["test"])


if __name__ == "__main__":
    unittest.main()
