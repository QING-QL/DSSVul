# SPDX-License-Identifier: Apache-2.0
# Backbone adaptations: LineVul, ReVeal and DGL; see THIRD_PARTY_NOTICES.md.
# Modified for DSSVul: normalized embeddings, shared heads and PyTorch graph aggregation.

import torch
from torch import nn
from torch.nn import functional as F


class Detector(nn.Module):
    def forward(self, inputs):
        return self.decode(self.encode(inputs))


class FeatureMLP(Detector):

    def __init__(self, input_dim, hidden=128):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.head = nn.Linear(hidden, 2)

    def encode(self, x):
        return F.normalize(self.encoder(x), dim=-1)

    def decode(self, z):
        return self.head(z)


class ASTGGNN(Detector):

    def __init__(self, num_types, hidden=128, steps=6):
        super().__init__()
        from torch_geometric.nn import GatedGraphConv
        self.embedding = nn.Embedding(num_types, hidden)
        self.ggnn = GatedGraphConv(hidden, num_layers=steps)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def encode(self, batch):
        from torch_geometric.nn import global_mean_pool
        x = self.ggnn(self.embedding(batch.x_type), batch.edge_index)
        return F.normalize(global_mean_pool(x, batch.batch), dim=-1)

    def decode(self, z):
        return self.head(z)


class LineVul(Detector):


    def __init__(self, model_path, normalize=True):
        super().__init__()
        from transformers import RobertaModel
        self.encoder = RobertaModel.from_pretrained(model_path, local_files_only=True,
                                                   add_pooling_layer=False)
        cfg = self.encoder.config
        self.normalize = normalize
        self.head = nn.Sequential(nn.Dropout(cfg.hidden_dropout_prob),
                                  nn.Linear(cfg.hidden_size, cfg.hidden_size), nn.Tanh(),
                                  nn.Dropout(cfg.hidden_dropout_prob), nn.Linear(cfg.hidden_size, 2))

    def encode_raw(self, inputs):
        return self.encoder(**inputs).last_hidden_state[:, 0]

    def encode(self, inputs):
        z = self.encode_raw(inputs)
        return F.normalize(z, dim=-1) if self.normalize else z

    def forward_original(self, inputs):

        return self.decode(self.encode_raw(inputs))

    def decode(self, z):
        return self.head(z)


class ReVealFeatures(Detector):


    def __init__(self, input_dim, hidden=256, layers=1, dropout=0.2):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.feature = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, hidden), nn.ReLU(), nn.Dropout(dropout)) for _ in range(layers)])
        self.head = nn.Linear(hidden, 2)

    def encode(self, x):
        x = self.layer1(x)
        for layer in self.feature:
            x = layer(x)
        return F.normalize(x, dim=-1)

    def decode(self, z):
        return self.head(z)


class TypedGGNN(nn.Module):


    def __init__(self, input_dim, hidden, relations, steps=6):
        super().__init__()
        if not 0 < input_dim <= hidden or relations < 1 or steps < 1:
            raise ValueError("Require 0 < input_dim <= hidden, relations/steps > 0")
        self.hidden, self.steps = hidden, steps
        self.messages = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(relations)])
        self.update = nn.GRUCell(hidden, hidden)
        for layer in self.messages:
            nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain("relu"))
            nn.init.zeros_(layer.bias)

    def forward(self, x, edge_index, edge_type):
        if edge_index.shape != (2, len(edge_type)):
            raise ValueError("edge_index/edge_type shape mismatch")
        if edge_type.numel() and (edge_type.min() < 0 or edge_type.max() >= len(self.messages)):
            raise ValueError("Unknown edge type")
        h = F.pad(x, (0, self.hidden - x.shape[1]))
        relation_edges = [edge_index[:, edge_type == k] for k in range(len(self.messages))]
        for _ in range(self.steps):
            incoming = torch.zeros_like(h)
            for layer, edges in zip(self.messages, relation_edges):
                if edges.shape[1]:
                    incoming.index_add_(0, edges[1], layer(h[edges[0]]))
            h = self.update(incoming, h)
        return h


class ReVeal(Detector):


    def __init__(self, input_dim, relations, graph_hidden=200, hidden=256,
                 steps=6, layers=1, dropout=0.2, normalize=True):
        super().__init__()
        self.ggnn = TypedGGNN(input_dim, graph_hidden, relations, steps)
        self.representation = ReVealFeatures(graph_hidden, hidden, layers, dropout)
        self.normalize = normalize

    def encode_raw(self, batch):
        from torch_geometric.nn import global_add_pool
        h = self.ggnn(batch.x, batch.edge_index, batch.edge_type)
        z = self.representation.layer1(global_add_pool(h, batch.batch))
        for layer in self.representation.feature:
            z = layer(z)
        return z

    def encode(self, batch):
        z = self.encode_raw(batch)
        return F.normalize(z, dim=-1) if self.normalize else z

    def decode(self, z):
        return self.representation.head(z)

    def forward_original(self, batch):
        return self.decode(self.encode_raw(batch))
