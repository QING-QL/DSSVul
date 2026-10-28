import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, matthews_corrcoef, precision_recall_fscore_support


def metrics(y, prediction, scores=None):
    y = np.asarray(y)
    prediction = np.asarray(prediction)
    if y.ndim != 1 or prediction.shape != y.shape or len(y) == 0:
        raise ValueError("Labels and predictions must be nonempty aligned vectors")
    if not np.isin(y, [0, 1]).all() or not np.isin(prediction, [0, 1]).all():
        raise ValueError("Labels and predictions must use 0/1 classes")
    p, r, f, _ = precision_recall_fscore_support(y, prediction, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    result = dict(precision=float(p), recall=float(r), f1=float(f),
                  mcc=float(matthews_corrcoef(y, prediction)), tn=int(tn), fp=int(fp),
                  fn=int(fn), tp=int(tp), n=len(y))
    if scores is not None:
        scores = np.asarray(scores)
        if scores.shape != y.shape or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError("Vulnerable-class scores must be aligned finite probabilities")
        result["auprc"] = float(average_precision_score(y, scores))
    return result
