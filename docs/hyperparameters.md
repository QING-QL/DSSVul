# Hyperparameters

Training parameters are set in the JSON files under `configs/`. The table lists the values in the supplied JSON configurations. Values specified in these files override the training code's built-in defaults; omitted fields use the built-in defaults. Save a separate configuration and use a new output directory for each experiment.

| Parameter | Configuration key | Supplied configuration value |
| --- | --- | --- |
| Labeled fraction of training data | `labeled_fraction` | 0.1 |
| Logical batch size | `batch_size` | 1024 |
| Micro-batch size | `micro_batch_size` | 4 |
| Maximum epochs per teacher/student stage | `teacher_epochs`, `student_epochs` | 3000 |
| AdamW learning rate | `lr` | 0.001 |
| Weight decay | `weight_decay` | 0.00001 |
| Focal weights `[non-vulnerable, vulnerable]` | `alpha` | `[0.1, 0.9]` |
| Focal focusing parameter | `gamma` | 2 |
| Gaussian perturbation standard deviation | `delta` | 0.01 |
| Perturbation count S | `perturbations` | 20 |
| Cluster count K | `clusters` | 10 |
| Balance weight lambda | `balance_weight` | 0.7 |
| Log-prior scale mu | `mu` | 1.0 |
| NRCE bottom fraction | `nrce_fraction` | 0.1 |
| Maximum self-training rounds | `max_rounds` | 10 |
| Epoch patience | `patience` | 30 |
| Validation selection metric | `selection_metric` | `mcc` |
| Token length | `max_length` | 512 |

The labeled fraction, logical batch size, epoch cap, optimizer, learning rate and weight decay follow the experimental protocol. Select dataset-specific parameter values using validation data and fix them before test evaluation. The paper's sensitivity-study settings are listed below.

Focal `alpha` is a pair ordered by class. The scalar vulnerable-class weight 0.9 corresponds to `[0.1, 0.9]`. The optional value `inverse_frequency` computes `[1 - n0/N, 1 - n1/N]` from `D_L` only.

The paper's sensitivity study reports F1-preferred settings of `(alpha_v, gamma, S, K, mu, lambda)` as `(0.90, 2, 20, 30, 1.25, 0.7)` for Devign, `(0.85, 2, 20, 30, 1.0, 0.5)` for ReVeal and `(0.90, 2, 30, 50, 1.75, 0.9)` for Big-Vul. These are sensitivity-study selections; they do not define a universal setting for the additional datasets or every backbone. In particular, ReVeal's reported MCC preference for `S` differs from its F1 preference. Choose the selection metric in advance.

Changing the training seed does not regenerate a supplied split manifest. The repeated-run command holds train/validation/test and the labeled subset fixed while varying training randomness. To study a different labeled budget or new split, prepare and identify a separate dataset partition. Rebuild graph features whenever `D_L` changes because the Word2Vec and node-type vocabulary are fitted on `D_L`.

Every new run writes its resolved configuration and software versions next to its checkpoint. These files belong to the local run directory and are excluded from the repository.
