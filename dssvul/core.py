from dataclasses import dataclass
import math

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.cluster import KMeans


def focal_loss(logits, targets, alpha, gamma=2.0, reduction="mean"):
    log_pt = F.log_softmax(logits, -1).gather(1, targets[:, None]).squeeze(1)
    loss = -alpha[targets] * (1 - log_pt.exp()).pow(gamma) * log_pt
    return loss.mean() if reduction == "mean" else loss


def nrce_loss(logits, targets, bottom_fraction=0.1, reduction="mean", cutoff=None):


    if not 0 < bottom_fraction <= 1:
        raise ValueError("bottom_fraction must be in (0, 1]")
    log_pt = F.log_softmax(logits, -1).gather(1, targets[:, None]).squeeze(1)
    pt = log_pt.exp()
    rank = max(1, math.ceil(len(pt) * bottom_fraction))
    cutoff = pt.detach().kthvalue(rank).values if cutoff is None else cutoff.detach()
    loss = torch.where(pt.detach() <= cutoff, -pt, -log_pt)
    return loss.mean() if reduction == "mean" else loss


def balance_loss(logits, targets, eligible, prior, mu=1.0, weight=1.0):


    if len(targets) == 0:
        return logits.sum() * 0
    adjusted = logits + mu * prior.log().to(logits)
    return weight * (F.cross_entropy(adjusted, targets, reduction="none") * eligible).mean()


def estimate_prior(probabilities, selected, classes=2):

    if len(probabilities) == 0:
        return 1.0, torch.full((classes,), 1 / classes), torch.zeros(0, dtype=torch.bool)
    confidence, prediction = probabilities.max(1)
    tau = confidence.mean().item()
    eligible = selected & (confidence >= tau)
    counts = torch.bincount(prediction[eligible], minlength=classes).float()
    prior = (counts + 1) / (counts.sum() + classes)
    return tau, prior, eligible


@torch.no_grad()
def entropy_confidence(model, embedding, samples=10, delta=0.01, generator=None):

    if samples < 1 or not 0 <= delta < 0.05:
        raise ValueError("samples >= 1 and 0 <= delta < 0.05 required")
    if model.training:
        raise ValueError("Use eval mode so uncertainty comes only from Gaussian noise")
    embedding = F.normalize(embedding, dim=-1)
    entropy = torch.zeros(len(embedding), device=embedding.device)
    classes = None
    for _ in range(samples):
        noise = torch.randn(embedding.shape, device=embedding.device,
                            dtype=embedding.dtype, generator=generator) * delta
        log_p = F.log_softmax(model.decode(embedding + noise), dim=-1)
        classes = log_p.shape[-1]
        entropy += -(log_p.exp() * log_p).sum(-1)
    entropy /= samples
    return (1 - entropy / math.log(classes)).clamp(0, 1)


@dataclass
class Selection:
    classifier: torch.Tensor
    cluster: torch.Tensor
    selected: torch.Tensor
    confidence_weight: torch.Tensor
    eligible: torch.Tensor
    prior: torch.Tensor
    tau: float
    cluster_count: int


def align_labels(z_unlabeled, probabilities, weights, z_labeled, labeled_targets,
                 clusters=10, seed=42, use_alignment=True):

    if len(z_unlabeled) != len(probabilities) or len(weights) != len(probabilities):
        raise ValueError("Unlabeled rows must align")
    if set(labeled_targets.tolist()) != {0, 1}:
        raise ValueError("Both classes are required in the labeled set for centroids")
    classifier = probabilities.argmax(1)
    if len(classifier) == 0:
        mask = torch.zeros(0, dtype=torch.bool)
        return Selection(classifier, classifier.clone(), mask, weights, mask,
                         torch.tensor([0.5, 0.5]), 1.0, 0)
    actual_k = 0
    if use_alignment:
        if clusters <= 2:
            raise ValueError("Paper protocol requires k > number of classes (2)")
        weighted = (1 + weights[:, None]) * F.normalize(z_unlabeled, dim=-1)
        unique_count = len(torch.unique(weighted, dim=0))
        if unique_count < clusters:
            raise ValueError(
                f"Cannot form the configured {clusters} clusters from "
                f"{unique_count} distinct confidence-scaled embeddings "
                f"({len(weighted)} samples). Inspect collapsed features or the "
                "configured K; the method does not silently reduce K."
            )
        km = KMeans(n_clusters=clusters, random_state=seed, n_init=10)
        assignments = km.fit_predict(weighted.numpy())
        actual_k = len(np.unique(assignments))
        if actual_k != clusters:
            raise ValueError(f"KMeans produced {actual_k} occupied clusters; expected K={clusters}")
        labeled = F.normalize(z_labeled, dim=-1)
        centroids = torch.stack([labeled[labeled_targets == c].mean(0) for c in range(2)])
        cluster_classes = torch.cdist(torch.as_tensor(km.cluster_centers_), centroids).argmin(1)
        cluster = cluster_classes[torch.as_tensor(assignments)]
        selected = cluster == classifier
    else:
        cluster = torch.full_like(classifier, -1)
        selected = torch.ones_like(classifier, dtype=torch.bool)
    tau, prior, eligible = estimate_prior(probabilities, selected)
    return Selection(classifier, cluster, selected, weights, eligible, prior, tau, actual_k)
