"""Separate supervised, topology-aware BA-2Motif reference experiment."""

from math import ceil

import networkx as nx
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import to_dense_adj

from cider import DEFAULT_RETAIN_RATIOS, undirected_edge_groups


def _triangle_count(x, edge_index, batch=None, edge_weight=None):
    if batch is None:
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
    if edge_weight is None:
        edge_weight = x.new_ones(edge_index.size(1))
    adjacency = to_dense_adj(edge_index, batch=batch, edge_attr=edge_weight)
    diagonal = 1 - torch.eye(
        adjacency.size(-1), dtype=adjacency.dtype, device=adjacency.device
    )
    adjacency = adjacency * diagonal.unsqueeze(0)
    adjacency = (adjacency + adjacency.transpose(1, 2)) * 0.5
    return torch.einsum("bij,bjk,bki->b", adjacency, adjacency, adjacency) / 6


def _edge_features(edge_index, num_nodes):
    pairs, inverse = undirected_edge_groups(edge_index)
    pair_list = pairs.cpu().tolist()
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from((u, v) for u, v in pair_list if u != v)
    bridges = {tuple(sorted(edge)) for edge in nx.bridges(graph)}
    features = [
        (
            u != v and (u, v) not in bridges,
            u != v and bool(set(graph[u]) & set(graph[v])),
            u == v,
        )
        for u, v in pair_list
    ]
    return torch.tensor(features, dtype=torch.float, device=edge_index.device).reshape(
        -1, 3
    )[inverse]


class BA2MotifModel(nn.Module):
    """Triangle graph classifier and supervised valid/priority edge heads."""

    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(1, 2)
        self.valid_head = nn.Linear(3, 1)
        self.priority_head = nn.Linear(3, 1)

    def forward(self, x, edge_index, batch=None, edge_weight=None):
        return self.classifier(
            _triangle_count(x, edge_index, batch, edge_weight).unsqueeze(-1)
        )

    def edge_logits(self, edge_index, num_nodes):
        features = _edge_features(edge_index, num_nodes)
        return (
            self.valid_head(features).squeeze(-1),
            self.priority_head(features).squeeze(-1),
        )

    def training_loss(self, data, *, valid_pos_weight=None, priority_pos_weight=None):
        valid, priority = self.edge_logits(data.edge_index, data.num_nodes)
        return (
            F.cross_entropy(self(data.x, data.edge_index, data.batch), data.y)
            + F.binary_cross_entropy_with_logits(
                valid, data.edge_valid, pos_weight=valid_pos_weight
            )
            + F.binary_cross_entropy_with_logits(
                priority, data.edge_priority, pos_weight=priority_pos_weight
            )
        )

    @torch.no_grad()
    def edge_scores(self, edge_index, num_nodes):
        valid, priority = self.edge_logits(edge_index, num_nodes)
        return (valid >= 0).float() * 2 + priority.sigmoid()

    @torch.no_grad()
    def explain_graph(self, data):
        scores = self.edge_scores(data.edge_index, data.num_nodes)
        pairs, inverse = undirected_edge_groups(data.edge_index)
        sums = scores.new_zeros(len(pairs))
        counts = scores.new_zeros(len(pairs))
        sums.scatter_add_(0, inverse, scores)
        counts.scatter_add_(0, inverse, torch.ones_like(scores))
        pair_scores = sums / counts
        explanations = {}
        for ratio in DEFAULT_RETAIN_RATIOS:
            count = min(len(pairs), max(1, ceil(len(pairs) * ratio)))
            keep = torch.zeros(len(pairs), dtype=torch.bool, device=scores.device)
            keep[pair_scores.topk(count).indices] = True
            mask = keep[inverse]
            graph = data.clone()
            graph.edge_index = data.edge_index[:, mask]
            if data.edge_attr is not None:
                graph.edge_attr = data.edge_attr[mask]
            explanations[format(ratio, ".10g")] = graph
        return explanations
