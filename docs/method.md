# Method

The implementation uses `0` for non-vulnerable functions and `1` for vulnerable functions. `D_L` contains labeled training functions, `D_U` contains the remaining training functions, and `D_S` contains agreement-selected pseudo-labeled functions. Validation and test functions never enter `D_U`.

## Teacher and pseudo-labels

1. Train the teacher on `D_L` with alpha-balanced focal loss: `-alpha[y] * (1 - p[y])**gamma * log(p[y])`.
2. Normalize each encoder embedding to unit length. Add independent Gaussian noise with standard deviation `delta` for each of `S` classifier evaluations. The classifier runs in evaluation mode.
3. Compute the average predictive entropy and `s = 1 - mean_entropy / log(2)`. Scale each unlabeled embedding by `1 + s` without renormalizing it.
4. Apply K-means with the configured `K`. Assign each cluster to the nearest labeled class centroid, using Euclidean distance. Labeled centroids are means of unit embeddings.
5. Retain functions whose cluster class matches the teacher's predicted class.

K-means uses ten initializations. `K` must exceed two and must not exceed the number of distinct scaled embeddings. An invalid clustering input raises an error instead of changing `K`.

## Student losses

The confidence threshold `tau` is the mean maximum teacher probability over all of `D_U`. The class prior is estimated from the members of `D_S` whose maximum teacher probability is at least `tau`, with one Laplace count per class.

Balance loss uses `logits + mu * log(prior)` and positive cross-entropy. Its sum is weighted by `lambda` and divided by the size of all `D_S`, including selected samples below `tau`. Those below-threshold samples contribute zero to this term.

NRCE is applied to all of `D_L` and `D_S`. Let `p_t` be the student's probability for the supplied ground-truth or pseudo-label. The cutoff is the `ceil(0.1 * B)`-th smallest `p_t` in the logical batch. For `p_t <= cutoff`, the loss is `-p_t`; otherwise it is `-log(p_t)`. The cutoff is detached from autograd. This piecewise definition is used directly, including its discontinuity.

Logical batches can be split into micro-batches. A first forward pass computes the NRCE cutoff across the entire logical batch. A second pass replays the random states and accumulates gradients. The balance contribution is scaled by `|D_L + D_S| / |D_S|` so uniform mixed-batch sampling estimates the stated `D_S`-normalized objective.

The student objective is NRCE plus balance loss. If `D_S` is empty, balance loss is zero and the student trains on `D_L` with NRCE.

## Model selection

Each stage keeps its best validation checkpoint and uses epoch-level patience. The student is initialized from the current teacher. Teacher promotion requires a strictly higher validation score; a rejected student does not stop the round loop. `min_delta` applies only to epoch-level checkpoint selection. The final test evaluation uses unadjusted model logits.

## Ablations

| Variant | Change |
| --- | --- |
| `full` | Complete method |
| `nce` | No perturbation uncertainty and no embedding scaling |
| `ndp` | Accept teacher pseudo-labels without clustering agreement |
| `nbl` | Student uses NRCE only; teacher still uses focal loss |
| `nnrce` | Replace student NRCE with ordinary cross-entropy |

## Backbones

LineVul uses CodeBERT's CLS representation and its dropout/dense/tanh classification head. ReVeal uses relation-specific GGNN messages, GRU updates, sum pooling and a representation MLP. Both expose a normalized encoder representation and a classifier. Training uses the DSSVul objectives above.
