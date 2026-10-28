# Third-party notices

DSSVul contributions: Copyright 2026 DSSVul contributors. Licensed under the Apache License, Version 2.0; see [LICENSE](LICENSE).

The backbone adapters in `dssvul/models.py` follow the upstream implementations listed below. Their original notices and license terms are retained. DSSVul adds a shared encoder/decoder interface, normalized representations for self-training, and native PyTorch graph aggregation. The adapters are trained using the teacher and student objectives described in [Method](docs/method.md).

| Component | Upstream source | License and notice |
| --- | --- | --- |
| LineVul classification head | [LineVul model](https://github.com/awsm-research/LineVul/blob/main/linevul/linevul_model.py) | [MIT](licenses/LineVul-MIT.txt), Copyright (c) 2022 anon-ai-research |
| ReVeal representation network | [ReVeal models](https://github.com/VulDetProject/ReVeal/blob/master/Vuld_SySe/representation_learning/models.py) | [MIT](licenses/ReVeal-MIT.txt), Copyright (c) 2021 VulDetProject |
| Typed gated graph updates and initialization | [DGL GatedGraphConv](https://github.com/dmlc/dgl/blob/master/python/dgl/nn/pytorch/conv/gatedgraphconv.py) | [Apache-2.0](licenses/DGL-Apache-2.0.txt) |

The graph encoder follows gated message passing with graph-level sum pooling, as in the [GGNNSum architecture linked by ReVeal](https://github.com/saikat107/Devign/blob/master/modules/model.py). DSSVul implements graph aggregation with PyTorch tensors and uses PyTorch Geometric for batching and pooling. The Devign repository and DGL runtime are not bundled.

LineVul uses the separately downloaded [CodeBERT](https://github.com/microsoft/CodeBERT) backbone. The [CodeBERT repository license](licenses/CodeBERT-MIT.txt), Copyright (c) Microsoft Corporation., is included for reference. Pretrained weights are obtained from the pinned model revision by `scripts/download_backbone.py` and are not included in this source package.

Other dependencies are installed from their own distributions. Dataset source links and processing metadata are documented in [Data preparation](docs/data.md); this repository's license does not relicense external datasets or model weights.
