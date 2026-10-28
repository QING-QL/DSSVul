# Data preparation

The task is function-level binary classification: `0` denotes non-vulnerable and `1` denotes vulnerable. Download source data from the authors and retain the original files. The processor does not remove code comments, alter whitespace, relabel functions or filter by a selected set of CWEs.

## Original sources

| Dataset | Author-maintained source | Download |
| --- | --- | --- |
| Devign | [Project](https://sites.google.com/view/devign/home) | [Function dataset](https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF/view) |
| ReVeal | [Repository](https://github.com/VulDetProject/ReVeal) | [Chromium and Debian data](https://drive.google.com/drive/folders/1KuIYgFcvWUXheDhT--cBALsfy1I4utOy) |
| Big-Vul | [Repository](https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset) | [Cleaned function CSV](https://drive.google.com/file/d/1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X/view) |
| PrimeVul | [Repository](https://github.com/DLVulDet/PrimeVul) | [Original release](https://drive.google.com/drive/folders/19iLaNDS0z99N8kB_jBRTmDLehwZBolMY), [v0.1](https://drive.google.com/drive/folders/1cznxGme5o6A_9tT8T47JUh3MPEpRYiKK) |
| DiverseVul | [Repository](https://github.com/wagner-group/diversevul) | [Function data](https://drive.google.com/file/d/12IWKhmLhq7qn5B_iXgn5YerOQtkH-6RG/view) |
| MegaVul | [Repository](https://github.com/Icyrockton/MegaVul) | [Author's release directory](https://1drv.ms/f/s!AtzrzuojQf5sgeISZ9zN_4owVnUn9g) |

Devign, ReVeal and Big-Vul are provided as source links. The processor can create new class-stratified 8:1:1 partitions for those downloaded inputs. Such partitions are identified by their generated manifests; the repository does not include the sample-ID lists used in the original experiments for those three datasets.

## Fixed inputs for the additional datasets

The compressed manifests in `datasets/splits/` contain sample hashes, row order, partition indices and class counts. They contain no function bodies or model results.

| Dataset | Vulnerable | Non-vulnerable | Total | Partition policy |
| --- | ---: | ---: | ---: | --- |
| PrimeVul | 6,004 | 218,529 | 224,533 | Preserve source train/validation/test |
| DiverseVul | 18,945 | 311,546 | 330,491 | Repository/commit group split |
| MegaVul | 17,449 | 323,532 | 340,981 | Repository/commit group split |

These counts identify the function sets used by the supplied manifests, rather than every upstream release. PrimeVul's original release and v0.1 differ in their sample coverage. The processor resolves the supplied identities and rejects missing samples or different labels.

The PrimeVul snapshot was adapted from the [colin/PrimeVul mirror](https://huggingface.co/datasets/colin/PrimeVul) with the source partitions retained. The supplied function hashes identify its exact membership. They also permit checking an author-hosted download without relying on the mirror's filename or row order.

The MegaVul input snapshot was obtained from the [CoVulPecker-Megavul mirror](https://huggingface.co/datasets/tronganhwork/CoVulPecker-Megavul/tree/b76e218ee868b1b5efcd1074c7f2fc71f11a2eb5), revision `b76e218ee868b1b5efcd1074c7f2fc71f11a2eb5`, file `megavul.json`, SHA256 `eabe8d3325be358d7a93327e3ae4f001725517a0067c4d80198f89904177419f`. The author-maintained link above remains the original source. The recorded hash identifies the mirror snapshot; it does not assert byte identity with the author's cloud release.

## Processing rules

1. Read CSV, Parquet, JSON arrays or JSON Lines. JSON Lines is detected from content even if its filename ends in `.json`.
2. Extract the function and binary label. MegaVul positive examples use `func_before`; negative examples use `func`. Big-Vul uses `func_before` with `vul`. Devign, PrimeVul and DiverseVul use `func` with `target`. Canonical CSV inputs use `code` and `target`; custom columns can be specified explicitly.
3. Exclude empty functions. Reject invalid labels. Remove all exact-code groups with conflicting labels and keep the first copy of remaining same-label duplicates. Every removal is recorded locally.
4. Identify the unchanged UTF-8 function body by SHA256. When a fixed manifest is supplied, select and order its exact sample identities and verify the label fingerprint. Additional source rows are listed as `outside_fixed_manifest`. Missing or conflicting required samples stop processing.
5. Preserve the supplied partitions. Without a manifest, PrimeVul retains its source partitions. DiverseVul and MegaVul use `StratifiedGroupKFold`: ten folds to select test, then nine folds on the remainder to select validation. The group key is project/repository plus commit, falling back to exact code identity. Other datasets use stratified random 8:1:1 splits.
6. Select approximately 10% of training functions as `D_L` by stratified sampling, with the remainder in `D_U`. All four partitions are disjoint and cover the canonical corpus.

For the fixed snapshots, DiverseVul excludes one empty non-vulnerable function. MegaVul excludes 300 records in 143 conflicting-label groups and 12,592 additional same-label duplicate records. PrimeVul preserves source assignments. Its repository/commit groups may span partitions; validation checks source membership without reassigning those functions. Each dataset is evaluated separately. Cross-dataset experiments require an additional overlap check because the sources share functions.

## Commands

Prepare PrimeVul from the source partition files:

```bash
python scripts/prepare_dataset.py --dataset primevul --inputs data/raw/primevul_train.jsonl data/raw/primevul_valid.jsonl data/raw/primevul_test.jsonl --source-splits train validation test --splits datasets/splits/primevul_seed42.json.gz --output data/primevul
```

Prepare DiverseVul and MegaVul:

```bash
python scripts/prepare_dataset.py --dataset diversevul --inputs data/raw/diversevul_20230702.json --splits datasets/splits/diversevul_seed42.json.gz --output data/diversevul
python scripts/prepare_dataset.py --dataset megavul --inputs data/raw/megavul.json --splits datasets/splits/megavul_seed42.json.gz --output data/megavul
```

The canonical CSV has `sample_id`, `code`, `target`, `project`, `commit_id`, `source_split`, `source_file` and `source_row`. Local manifests record input checksums and exclusions. Verify an existing prepared input with:

```bash
python scripts/verify_dataset.py --dataset diversevul --data data/diversevul/canonical.csv --splits data/diversevul/splits_seed42.json
```

To prepare a new partition from another source version, omit `--splits` and specify `--seed` and `--fraction`. Use a different output directory. A new partition is a separate experiment, not a substitute for a supplied fixed partition.

## Graph inputs

Install Joern separately and use its C/C++ frontend. Record the Joern version used. The graph pipeline consumes the same canonical CSV and manifest as the token pipeline:

```bash
python scripts/build_cpg.py --dataset diversevul --data data/diversevul/canonical.csv --splits data/diversevul/splits_seed42.json --output data/diversevul_cpg --joern /path/to/joern-cli --extension cpp
```

The exporter retains AST, CFG, REACHING_DEF and CDG edges and adds explicit reverse relations. Node features combine token Word2Vec means with one-hot node types. Word2Vec and the type vocabulary are fitted only on `D_L`; unknown node types use an unknown-type entry. Feature identities and encoder hashes are saved with the graph data.

Graph preparation keeps the original sample order and partition membership. It stops if a required function has no graph and writes the missing identities for inspection. It never substitutes a smaller test set. Full graph extraction depends on the selected Joern version and source syntax.

The prepared graph input is `data/diversevul_cpg/prepared/graphs.pt`, with its corresponding `splits.json`. Load graph and checkpoint files only from a trusted source because they use PyTorch serialization.
