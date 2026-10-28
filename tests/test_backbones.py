import copy
import unittest
from unittest.mock import patch
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data, Batch
from dssvul.models import TypedGGNN, ReVeal, LineVul
from dssvul.data import Corpus


class BackboneTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(24)

    def test_typed_ggnn_matches_independent_edge_loop_and_gradients(self):
        model = TypedGGNN(3, 5, 3, steps=2).double()
        reference = copy.deepcopy(model)
        x = torch.randn(4, 3, dtype=torch.double, requires_grad=True)
        xx = x.detach().clone().requires_grad_()
        edges = torch.tensor([[0, 2, 1, 0], [1, 1, 2, 3]])
        kinds = torch.tensor([0, 1, 2, 0])
        actual = model(x, edges, kinds)
        h = F.pad(xx, (0, 2))
        for _ in range(2):
            aggregated = []
            for dst in range(len(x)):
                messages = [reference.messages[int(kinds[e])](h[int(edges[0, e])])
                            for e in range(len(kinds)) if int(edges[1, e]) == dst]
                aggregated.append(torch.stack(messages).sum(0) if messages else h.new_zeros(5))
            h = reference.update(torch.stack(aggregated), h)
        self.assertTrue(torch.allclose(actual, h, atol=1e-12))
        actual.square().sum().backward(); h.square().sum().backward()
        self.assertTrue(torch.allclose(x.grad, xx.grad, atol=1e-12))
        for p, q in zip(model.parameters(), reference.parameters()):
            self.assertTrue(torch.allclose(p.grad, q.grad, atol=1e-12))

    def test_relation_types_affect_output_and_edgeless_graph_supported(self):
        model = TypedGGNN(3, 5, 2)
        x = torch.randn(3, 3)
        edge = torch.tensor([[0, 1], [1, 2]])
        self.assertFalse(torch.allclose(model(x, edge, torch.tensor([0, 0])), model(x, edge, torch.tensor([1, 1]))))
        self.assertTrue(torch.isfinite(model(x, torch.empty(2, 0, dtype=torch.long), torch.empty(0, dtype=torch.long))).all())

    def test_reveal_batch_isolation_and_node_permutation_invariance(self):
        model = ReVeal(3, 2, graph_hidden=8, hidden=12, steps=2).eval()
        a = Data(x=torch.randn(3, 3), edge_index=torch.tensor([[0, 1], [1, 2]]), edge_type=torch.tensor([0, 1]))
        b = Data(x=torch.randn(2, 3), edge_index=torch.empty(2, 0, dtype=torch.long), edge_type=torch.empty(0, dtype=torch.long))
        joint = model(Batch.from_data_list([a, b]))
        separate = torch.cat([model(Batch.from_data_list([g])) for g in [a, b]])
        self.assertTrue(torch.allclose(joint, separate, atol=1e-6))
        permutation = torch.tensor([2, 0, 1]); inverse = permutation.argsort()
        aa = Data(x=a.x[permutation], edge_index=inverse[a.edge_index], edge_type=a.edge_type)
        self.assertTrue(torch.allclose(model(Batch.from_data_list([a])), model(Batch.from_data_list([aa])), atol=1e-6))
        model.train(); F.cross_entropy(model(Batch.from_data_list([a, b])), torch.tensor([0, 1])).backward()
        self.assertGreater(model.ggnn.update.weight_ih.grad.abs().sum().item(), 0)
        self.assertGreater(model.representation.head.weight.grad.abs().sum().item(), 0)

    def test_cpg_batch_hides_ground_truth(self):
        import numpy as np
        g = Data(x=torch.ones(2, 3), edge_index=torch.tensor([[0], [1]]), edge_type=torch.tensor([0]), y=torch.tensor([1]))
        corpus = Corpus([g], [1], ["sample"], "cpg", {})
        batch = corpus.batch(np.array([0]), "cpu")
        self.assertIsNone(batch.y)
        self.assertEqual(batch.edge_type.tolist(), [0])

    def test_linevul_original_path_matches_roberta_classification_head(self):
        from transformers import RobertaConfig, RobertaModel
        from transformers.models.roberta.modeling_roberta import RobertaClassificationHead
        cfg = RobertaConfig(vocab_size=40, hidden_size=16, num_attention_heads=2,
                            num_hidden_layers=1, intermediate_size=24, num_labels=2,
                            hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
        encoder = RobertaModel(cfg, add_pooling_layer=False)
        with patch("transformers.RobertaModel.from_pretrained", return_value=encoder):
            model = LineVul("unused").eval()
        head = RobertaClassificationHead(cfg).eval()
        head.dense.load_state_dict(model.head[1].state_dict())
        head.out_proj.load_state_dict(model.head[4].state_dict())
        inputs = dict(input_ids=torch.tensor([[0, 5, 6, 2], [0, 7, 2, 1]]),
                      attention_mask=torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]))
        expected = head(encoder(**inputs).last_hidden_state)
        self.assertTrue(torch.allclose(model.forward_original(inputs), expected, atol=1e-7))
        z = model.encode(inputs)
        self.assertTrue(torch.allclose(z.norm(dim=-1), torch.ones(2), atol=1e-6))
        self.assertTrue(torch.allclose(model(inputs), model.decode(z), atol=1e-7))
        F.cross_entropy(model(inputs), torch.tensor([0, 1])).backward()
        self.assertGreater(model.encoder.embeddings.word_embeddings.weight.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
