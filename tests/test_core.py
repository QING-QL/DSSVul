import copy
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dssvul.core import focal_loss, nrce_loss, balance_loss, estimate_prior, entropy_confidence, align_labels
from dssvul.data import Corpus, make_splits, save_splits
from dssvul.models import FeatureMLP
from dssvul.train import DEFAULTS, fit, seed_all


class CoreTests(unittest.TestCase):
    def test_focal_gamma_zero_matches_weighted_ce(self):
        logits = torch.tensor([[1., -1.], [0.5, 2.]], requires_grad=True)
        y = torch.tensor([0, 1]); alpha = torch.tensor([0.2, 0.8])
        expected = (F.cross_entropy(logits, y, reduction="none") * alpha[y]).mean()
        self.assertTrue(torch.allclose(focal_loss(logits, y, alpha, 0), expected))

    def test_nrce_piecewise_values_and_bounded_low_confidence_gradient(self):
        logits = torch.tensor([[0., -8.], [0., 1.], [0., 2.], [0., 3.]], requires_grad=True)
        y = torch.ones(4, dtype=torch.long)
        p = logits.softmax(-1)[:, 1]
        loss = nrce_loss(logits, y, 0.25, "none")
        self.assertAlmostEqual(loss[0].item(), -p[0].item())
        self.assertTrue(torch.allclose(loss[1:], -p[1:].log()))
        gradient = torch.autograd.grad(loss[0], logits, retain_graph=True)[0][0, 1].abs()
        ce_gradient = torch.autograd.grad(-p[0].log(), logits)[0][0, 1].abs()
        self.assertLess(gradient, ce_gradient / 100)

    def test_nrce_threshold_ties_and_singleton(self):
        z = torch.zeros(4, 2, requires_grad=True)
        self.assertTrue(torch.allclose(nrce_loss(z, torch.zeros(4, dtype=torch.long), reduction="none"), torch.full((4,), -0.5)))
        self.assertAlmostEqual(nrce_loss(z[:1], torch.tensor([0])).item(), -0.5)

    def test_prior_threshold_uses_all_unlabeled(self):
        p = torch.tensor([[0.95, 0.05], [0.4, 0.6], [0.2, 0.8], [0.45, 0.55]])
        selected = torch.tensor([True, True, False, False])
        tau, prior, mask = estimate_prior(p, selected)
        self.assertAlmostEqual(tau, 0.725, places=6)
        self.assertEqual(mask.tolist(), [True, False, False, False])
        self.assertTrue(torch.allclose(prior, torch.tensor([2/3, 1/3])))
        _, empty_prior, _ = estimate_prior(p, torch.zeros(4, dtype=torch.bool))
        self.assertTrue(torch.allclose(empty_prior, torch.tensor([0.5, 0.5])))

    def test_balance_denominator_and_correction_sign(self):
        z = torch.zeros(2, 2, requires_grad=True)
        prior = torch.tensor([0.8, 0.2])
        loss = balance_loss(z, torch.tensor([1, 0]), torch.tensor([True, False]), prior, 1, 0.7)
        self.assertAlmostEqual(loss.item(), 0.7 * -math.log(0.2) / 2, places=6)
        loss.backward()
        self.assertEqual(z.grad[1].abs().sum().item(), 0)
        self.assertLess(z.grad[0, 1].item(), 0)

    def test_entropy_zero_noise_and_average_entropy_definition(self):
        class Head(nn.Module):
            def decode(self, z):
                return z * 400
        model = Head().eval()
        z = torch.zeros(128, 2)
        self.assertTrue(torch.allclose(entropy_confidence(model, z, 3, 0), torch.zeros(128), atol=1e-6))


        weight = entropy_confidence(model, z, 100, 0.01, torch.Generator().manual_seed(1))
        self.assertGreater(weight.mean().item(), 0.55)
        self.assertTrue(((weight >= 0) & (weight <= 1)).all())

    def test_alignment_centroid_mapping_not_cluster_id_or_majority_vote(self):
        z_l = torch.tensor([[1., 0.], [-1., 0.]])
        y_l = torch.tensor([1, 0])
        z_u = torch.tensor([[1., 0.], [1., 0.1], [-1., 0.], [-1., 0.1]])
        p = torch.tensor([[0.1, 0.9], [0.9, 0.1], [0.9, 0.1], [0.1, 0.9]])
        s = align_labels(z_u, p, torch.ones(4), z_l, y_l, clusters=3)
        self.assertEqual(s.cluster.tolist(), [1, 1, 0, 0])
        self.assertEqual(s.selected.tolist(), [True, False, True, False])

    def test_empty_selection_and_collapsed_features_do_not_silently_change_k(self):
        labeled = torch.tensor([[1., 0.], [-1., 0.]])
        y = torch.tensor([0, 1])
        s = align_labels(torch.empty(0, 2), torch.empty(0, 2), torch.empty(0), labeled, y)
        self.assertEqual(len(s.selected), 0)
        with self.assertRaisesRegex(ValueError, "does not silently reduce K"):
            align_labels(torch.ones(4, 2), torch.tensor([[0.9, 0.1]] * 4), torch.ones(4), labeled, y)

    def test_alignment_preserves_configured_cluster_count(self):
        z = torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
        p = torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]])
        result = align_labels(z, p, torch.ones(4), z[[0, 2]], torch.tensor([0, 1]), clusters=3)
        self.assertEqual(result.cluster_count, 3)
        with self.assertRaisesRegex(ValueError, "Cannot form the configured 5 clusters"):
            align_labels(z, p, torch.ones(4), z[[0, 2]], torch.tensor([0, 1]), clusters=5)

    def test_splits_disjoint_and_stable(self):
        corpus = Corpus(torch.randn(200, 4), [0] * 160 + [1] * 40, range(200), "features", {})
        split = make_splits(corpus, 0.1, 7)
        self.assertEqual({k: len(v) for k, v in split.items()}, dict(labeled=16, unlabeled=144, validation=20, test=20))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "split.json"
            save_splits(path, corpus, split)
            restored = make_splits(corpus, manifest=path)
            self.assertTrue(all(np.array_equal(split[k], restored[k]) for k in split))
            corpus.ids[0] = "changed"
            with self.assertRaises(ValueError):
                make_splits(corpus, manifest=path)

    def test_graph_batch_does_not_expose_labels(self):
        from torch_geometric.data import Data
        g = Data(x_type=torch.tensor([0, 1]), edge_index=torch.tensor([[0], [1]]), y=torch.tensor([1]))
        corpus = Corpus([g], [1], ["a"], "graph", {})
        batch = corpus.batch(np.array([0]), "cpu")
        self.assertIsNone(batch.y)

    def test_graph_batch_preserves_isolated_nodes(self):
        from torch_geometric.data import Data
        g = Data(x_type=torch.tensor([0, 1, 2]), edge_index=torch.tensor([[0], [1]]))
        corpus = Corpus([g, g], [0, 1], ["a", "b"], "graph", {})
        batch = corpus.batch(np.array([0, 1]), "cpu")
        self.assertEqual(batch.batch.tolist(), [0, 0, 0, 1, 1, 1])
        self.assertEqual(batch.edge_index.tolist(), [[0, 3], [1, 4]])

    def test_two_pass_logical_batch_matches_direct_nrce_update_with_dropout(self):

        from dssvul.models import Detector
        class DropoutModel(Detector):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(nn.Linear(3, 5), nn.Dropout(0.3))
                self.head = nn.Linear(5, 2)
            def encode(self, x):
                return F.normalize(self.encoder(x), dim=-1)
            def decode(self, z):
                return self.head(z)
        seed_all(11)
        corpus = Corpus(torch.randn(12, 3), [0, 1] * 6, range(12), "features", {})
        base = DropoutModel()
        candidate = copy.deepcopy(base)
        config = {**DEFAULTS, "batch_size": 8, "micro_batch_size": 3,
                  "student_epochs": 1, "seed": 11, "inference_batch_size": 4}
        targets = corpus.labels[:8]
        pseudo = np.array([False] * 4 + [True] * 4)
        eligible = np.array([False] * 4 + [True, False, True, False])
        prior = torch.tensor([0.7, 0.3]); alpha = torch.tensor([0.5, 0.5])
        before_rng = torch.get_rng_state()
        opt = torch.optim.AdamW(base.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        order = np.random.default_rng(12).permutation(8)
        outputs = [base(corpus.inputs[order[i:i+3]]) for i in range(0, 8, 3)]
        logits = torch.cat(outputs)
        y = torch.tensor(targets[order])
        loss = nrce_loss(logits, y) + 2 * (F.cross_entropy(logits + prior.log(), y, reduction="none") * torch.tensor(eligible[order])).mean()
        loss.backward(); opt.step()
        torch.set_rng_state(before_rng)
        with tempfile.TemporaryDirectory() as d:
            fit(candidate, corpus, np.arange(8), targets, pseudo, eligible, prior, np.arange(8, 12),
                config, "student_1", Path(d), alpha)
        for p, q in zip(base.parameters(), candidate.parameters()):
            self.assertTrue(torch.allclose(p, q, atol=2e-7), (p-q).abs().max().item())


if __name__ == "__main__":
    unittest.main()
