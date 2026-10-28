# Evaluation metrics

Vulnerable functions are the positive class (`1`). Predictions are the argmax of the two unadjusted probabilities. An exact tie is assigned to class `0`. All test samples are collected before metrics are computed; batch-level MCC or F1 values are not averaged.

With true positives `TP`, true negatives `TN`, false positives `FP` and false negatives `FN`:

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 TP / (2 TP + FP + FN)
MCC       = (TP TN - FP FN) /
            sqrt((TP + FP)(TP + FN)(TN + FP)(TN + FN))
```

Undefined Precision, Recall or F1 denominators return zero. A zero MCC denominator returns zero, following `sklearn.metrics.matthews_corrcoef`. MCC ranges from -1 to 1; the other metrics range from 0 to 1. Stored values are not percentages. Multiplication by 100 is only a display conversion.

AUPRC is non-interpolated Average Precision, computed by `sklearn.metrics.average_precision_score` from the continuous probability of class `1`. It is the sum of each increase in recall times the precision at that point. It is not the trapezoidal area under a plotted PR curve and is not calculated from binary predictions.

Tests compare MCC, Precision, Recall and F1 with independent confusion-matrix formulas across 624 combinations, and cover perfect, reversed, constant and imbalanced predictions. Average Precision is checked against an independent rank-based calculation and tied-score examples. The standalone checkpoint evaluator uses the same metric calculation as training and is tested for consistency with the final training evaluation.

For repeated runs, calculate every metric separately for each seed, then report the arithmetic mean and sample standard deviation (`ddof=1`). Pooling predictions across seeds defines a different statistic.

Reference definitions: [MCC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html) and [Average Precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).
