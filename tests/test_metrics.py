import math
import unittest
import warnings
from itertools import product

import numpy as np

from dssvul.metrics import metrics


class MetricTests(unittest.TestCase):
    def test_invalid_inputs_rejected(self):
        for labels, predictions, scores in (([], [], None), ([0, 2], [0, 1], None),
                                             ([0, 1], [0], None), ([0, 1], [0, 1], [0.3, float("nan")])):
            with self.assertRaises(ValueError):
                metrics(labels, predictions, scores)

    def test_confusion_matrix_formulas(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for tn, fp, fn, tp in product(range(5), repeat=4):
                if tn + fp + fn + tp == 0:
                    continue
                truth = [0] * (tn + fp) + [1] * (fn + tp)
                predictions = [0] * tn + [1] * fp + [0] * fn + [1] * tp
                measured = metrics(truth, predictions)
                denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
                expected_mcc = (tp * tn - fp * fn) / denominator if denominator else 0.0
                expected_p = tp / (tp + fp) if tp + fp else 0.0
                expected_r = tp / (tp + fn) if tp + fn else 0.0
                expected_f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
                self.assertAlmostEqual(measured["mcc"], expected_mcc, places=14)
                self.assertAlmostEqual(measured["precision"], expected_p, places=14)
                self.assertAlmostEqual(measured["recall"], expected_r, places=14)
                self.assertAlmostEqual(measured["f1"], expected_f1, places=14)
                self.assertEqual([measured[k] for k in ("tn", "fp", "fn", "tp")], [tn, fp, fn, tp])

    def test_mcc_perfect_reversed_constant_and_imbalanced(self):
        y = [0] * 997 + [1] * 3
        self.assertEqual(metrics(y, y)["mcc"], 1.0)
        self.assertEqual(metrics(y, [1 - value for value in y])["mcc"], -1.0)
        self.assertEqual(metrics(y, [0] * len(y))["mcc"], 0.0)
        self.assertEqual(metrics(y, [1] * len(y))["mcc"], 0.0)
        y = [0] * 94 + [1] * 12
        prediction = [0] * 91 + [1] * 3 + [0] * 5 + [1] * 7
        self.assertAlmostEqual(metrics(y, prediction)["mcc"], (7 * 91 - 3 * 5) / math.sqrt(10 * 12 * 94 * 96))

    def test_ap_matches_rank_sum_without_ties(self):
        rng = np.random.default_rng(183)
        for count in (10, 31, 1000):
            truth = rng.integers(0, 2, count)
            truth[:2] = [0, 1]
            scores = rng.random(count)
            ranked = truth[np.argsort(-scores)]
            expected = sum(np.cumsum(ranked)[i] / (i + 1) for i in range(count) if ranked[i]) / sum(ranked)
            result = metrics(truth, (scores > 0.5).astype(int), scores)
            self.assertAlmostEqual(result["auprc"], expected, places=14)

    def test_ap_ties_and_threshold_independence(self):
        y = [1, 0, 1, 0]
        scores = [0.9, 0.8, 0.7, 0.1]
        self.assertAlmostEqual(metrics(y, [0] * 4, scores)["auprc"], (1 + 2 / 3) / 2)
        self.assertEqual(metrics(y, [0] * 4, scores)["auprc"], metrics(y, [1] * 4, scores)["auprc"])
        self.assertEqual(metrics(y, [0] * 4, [0.5] * 4)["auprc"], 0.5)
        self.assertAlmostEqual(metrics(y, [1, 1, 0, 0], [0.9, 0.9, 0.1, 0.1])["auprc"], 0.5)


if __name__ == "__main__":
    unittest.main()
