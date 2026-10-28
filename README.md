# DSSVul

Implementation of **DSSVul: Debiased Semi-supervised Self-training for Class-Imbalanced Software Vulnerability Detection**.

DSSVul trains a teacher with focal loss, selects pseudo-labels by agreement between teacher predictions and uncertainty-weighted clustering, and trains a student with distribution-aware balancing and noise-robust cross-entropy. LineVul and ReVeal are supported as backbones.

## Installation

Use Python 3.10 with the dependencies listed below. Install a PyTorch build suitable for your CUDA version, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The reference dependencies are listed in `requirements.txt`. CPU tests use small generated inputs and do not download pretrained weights:

```bash
python -m unittest discover -s tests -v
```

## Data

Download datasets from their original sources and follow [Data preparation](docs/data.md). The repository contains processing code and fixed sample-ID manifests for PrimeVul, DiverseVul and MegaVul. Dataset files are downloaded separately, and model outputs are generated locally.

Prepare DiverseVul using the supplied partition:

```bash
python scripts/prepare_dataset.py --dataset diversevul --inputs data/raw/diversevul_20230702.json --splits datasets/splits/diversevul_seed42.json.gz --output data/diversevul
```

The output contains `canonical.csv`, `splits_seed42.json`, a source manifest and an exclusion ledger. All configuration paths are relative to the repository root.

## Training

Download CodeBERT for the LineVul backbone:

```bash
python scripts/download_backbone.py --output pretrained/codebert-base
```

Check the data paths and device in the configuration, then train:

```bash
python -m dssvul --config configs/linevul_diversevul.json
```

For ReVeal, first prepare CPG features using the same functions and partitions. See [Graph inputs](docs/data.md#graph-inputs), then use `configs/reveal_diversevul.json`.

The training procedure executes the configured number of self-training rounds. Each student starts from the current teacher. A student replaces the teacher only when its validation score improves. The test set is evaluated after the last round.

Hyperparameters are stored in configuration files. See [Hyperparameters](docs/hyperparameters.md) for their meaning and selection protocol, and [Method](docs/method.md) for the loss definitions.

To repeat training on the same fixed partition with different initialization seeds:

```bash
python run_experiments.py --config configs/linevul_diversevul.json --output-root runs/diversevul --seeds 42 43 44 45 46 --variants full
```

Use `--plan-only` to print the configurations. Supported variants are `full`, `nce`, `ndp`, `nbl` and `nnrce`.

## Evaluation

Evaluate a checkpoint with the data and split file used for that run:

```bash
python -m dssvul.evaluate --checkpoint runs/linevul_diversevul_seed42/best.pt --data data/diversevul/canonical.csv --splits runs/linevul_diversevul_seed42/splits.json --output evaluations/linevul_diversevul_seed42 --device cuda:0
```

Evaluation loads the saved weights, checks the data and partition hashes, and computes Precision, Recall, F1, MCC and AUPRC over the complete test set. It does not train a model or regenerate partitions. See [Metrics](docs/metrics.md) for definitions and numerical conventions.

For one source file with a LineVul checkpoint:

```bash
python -m dssvul.predict --checkpoint runs/linevul_diversevul_seed42/best.pt --data example.c --output predictions/example.json --device cpu
```

If the pretrained backbone has moved, pass `--model-path` to evaluation or prediction. Use the same pretrained revision and tokenizer.

## References

The backbones follow [LineVul](https://github.com/awsm-research/LineVul) and [ReVeal](https://github.com/VulDetProject/ReVeal). LineVul uses [CodeBERT](https://github.com/microsoft/CodeBERT). Dataset links and version identifiers are listed in [Data preparation](docs/data.md).

## License

DSSVul contributions are licensed under [Apache-2.0](LICENSE). Upstream components retain their original licenses and copyright notices; see [Third-party notices](THIRD_PARTY_NOTICES.md). External datasets, pretrained weights and installed dependencies remain subject to their respective licenses.
