"""Recursive CIDER: model, training objective, and graph explanation."""

from math import ceil

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool


DEFAULT_RETAIN_RATIOS = tuple(value / 10 for value in range(1, 10))


def undirected_edge_groups(edge_index):
    """Map both directions of an undirected edge to one canonical pair."""
    canonical = torch.sort(edge_index, dim=0).values.t().contiguous()
    return torch.unique(canonical, dim=0, sorted=True, return_inverse=True)


def prediction_margin(logits, target):
    targets = torch.as_tensor(target, device=logits.device, dtype=torch.long).view(-1)
    if targets.numel() == 1:
        targets = targets.expand(logits.size(0))
    rows = torch.arange(logits.size(0), device=logits.device)
    other_logits = logits.clone()
    other_logits[rows, targets] = -torch.inf
    return logits[rows, targets] - other_logits.max(dim=1).values


class GcnTaskModel(nn.Module):
    """Graph classifier matching the official NCI1/Mutagenicity task weights."""

    def __init__(
        self,
        input_dim,
        hidden_dim=50,
        embedding_dim=10,
        label_dim=2,
        num_layers=3,
        pred_hidden_dims=(10, 10),
    ):
        super().__init__()
        self.label_dim = label_dim
        self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        self.conv_first = GCNConv(input_dim, hidden_dim)
        self.conv_block = nn.ModuleList(
            GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers - 2)
        )
        self.conv_last = GCNConv(hidden_dim, embedding_dim)
        layers = []
        current_dim = self.pred_input_dim
        for output_dim in pred_hidden_dims:
            layers.extend((nn.Linear(current_dim, output_dim), nn.ReLU()))
            current_dim = output_dim
        layers.append(nn.Linear(current_dim, label_dim))
        self.pred_model = nn.Sequential(*layers) if pred_hidden_dims else layers[0]

    def forward(self, x, edge_index, batch=None, edge_weight=None, edge_attr=None):
        x = F.relu(self.conv_first(x, edge_index, edge_weight=edge_weight))
        outputs = [x]
        for layer in self.conv_block:
            x = F.relu(layer(x, edge_index, edge_weight=edge_weight))
            outputs.append(x)
        outputs.append(self.conv_last(x, edge_index, edge_weight=edge_weight))
        nodes = torch.cat(outputs, dim=1)
        return self.pred_model(global_mean_pool(nodes, batch))


class SymmetricEdgeDecoder(nn.Module):
    def __init__(self, representation_channels, hidden_channels):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(representation_channels * 3 + 1, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, z, edge_index, retain_ratio):
        source, target = z[edge_index[0]], z[edge_index[1]]
        features = torch.cat(
            (source + target, (source - target).abs(), source * target), dim=-1
        )
        ratio = features.new_full((features.size(0), 1), retain_ratio)
        return self.network(torch.cat((features, ratio), dim=-1)).squeeze(-1)


class IterativeCIDER(nn.Module):
    """Learn small non-causal deletions and re-encode the residual each round."""

    LOSS_WEIGHTS = {
        "fidelity": 1.0,
        "distillation": 0.25,
        "margin": 0.25,
        "step_consistency": 0.5,
        "counterfactual": 1.0,
        "kl": 1e-5,
        "independence": 0.01,
        "invariance": 0.01,
        "entropy": 0.001,
    }

    def __init__(
        self,
        in_channels,
        task_model,
        *,
        hidden_channels=48,
        latent_channels=16,
        encoder_layers=2,
        decoder_channels=64,
        dropout=0.0,
        num_classes=None,
        retain_ratios=DEFAULT_RETAIN_RATIOS,
    ):
        super().__init__()
        self.task_model = task_model
        for parameter in task_model.parameters():
            parameter.requires_grad_(False)
        self.num_classes = (
            int(getattr(task_model, "label_dim", 2))
            if num_classes is None
            else num_classes
        )
        self.retain_ratios = tuple(sorted(set(retain_ratios), reverse=True))
        self.dropout = dropout
        channels = [in_channels] + [hidden_channels] * encoder_layers
        self.shared_convs = nn.ModuleList(
            GCNConv(channels[index], channels[index + 1])
            for index in range(encoder_layers)
        )
        self.shared_norms = nn.ModuleList(
            nn.LayerNorm(hidden_channels) for _ in range(encoder_layers)
        )
        self.mu_causal = GCNConv(hidden_channels, latent_channels)
        self.mu_non_causal = GCNConv(hidden_channels, latent_channels)
        self.logvar_causal = GCNConv(hidden_channels, latent_channels)
        self.logvar_non_causal = GCNConv(hidden_channels, latent_channels)
        representation_channels = (
            latent_channels + hidden_channels * 2 + in_channels + self.num_classes
        )
        self.causal_decoder = SymmetricEdgeDecoder(
            representation_channels, decoder_channels
        )
        self.non_causal_decoder = SymmetricEdgeDecoder(
            representation_channels, decoder_channels
        )
        for layer in (self.logvar_causal, self.logvar_non_causal):
            nn.init.zeros_(layer.lin.weight)
            nn.init.constant_(layer.bias, -3.0)

    @property
    def device(self):
        return next(self.shared_convs.parameters()).device

    def train(self, mode=True):
        super().train(mode)
        self.task_model.eval()
        return self

    def _task_logits(
        self, x, edge_index, batch=None, *, edge_weight=None, edge_attr=None
    ):
        kwargs = {"batch": batch, "edge_weight": edge_weight}
        if edge_attr is not None:
            kwargs["edge_attr"] = edge_attr
        return self.task_model(x, edge_index, **kwargs)

    def _encode_state(self, x, edge_index, edge_weight=None):
        hidden = x
        for convolution, normalization in zip(self.shared_convs, self.shared_norms):
            hidden = convolution(hidden, edge_index, edge_weight=edge_weight)
            hidden = F.silu(normalization(hidden))
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return (
            hidden,
            self.mu_causal(hidden, edge_index, edge_weight=edge_weight),
            self.mu_non_causal(hidden, edge_index, edge_weight=edge_weight),
            self.logvar_causal(hidden, edge_index, edge_weight=edge_weight).clamp(
                -8, 4
            ),
            self.logvar_non_causal(hidden, edge_index, edge_weight=edge_weight).clamp(
                -8, 4
            ),
        )

    @staticmethod
    def _group_mean(values, inverse, num_groups):
        sums = values.new_zeros(num_groups)
        counts = values.new_zeros(num_groups)
        sums.scatter_add_(0, inverse, values)
        counts.scatter_add_(0, inverse, torch.ones_like(values))
        return sums / counts.clamp_min(1)

    def _edge_distribution(
        self, mu_c, mu_n, context, edge_index, inverse, num_pairs, ratio
    ):
        logits = []
        for representation, decoder in (
            (torch.cat((mu_c, context), dim=-1), self.causal_decoder),
            (torch.cat((mu_n, context), dim=-1), self.non_causal_decoder),
        ):
            scores = decoder(representation, edge_index, ratio)
            logits.append(self._group_mean(scores, inverse, num_pairs))
        logits = torch.stack(logits, dim=-1)
        return logits, logits.softmax(dim=-1)

    def _node_context(self, hidden, x, batch, targets):
        graph_context = global_mean_pool(hidden, batch)[batch]
        target_context = F.one_hot(targets, num_classes=self.num_classes).to(x.dtype)[
            batch
        ]
        return torch.cat((hidden, x, graph_context, target_context), dim=-1)

    def forward(
        self,
        x,
        edge_index,
        *,
        edge_weight=None,
        retain_ratio=1.0,
        batch=None,
        target=None,
    ):
        pairs, inverse = undirected_edge_groups(edge_index)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        if target is None:
            with torch.no_grad():
                target = self._task_logits(x, edge_index, batch).argmax(dim=-1)
        targets = torch.as_tensor(target, dtype=torch.long, device=x.device).view(-1)
        if targets.numel() == 1:
            targets = targets.expand(int(batch.max().item()) + 1)
        hidden, mu_c, mu_n, logvar_c, logvar_n = self._encode_state(
            x, edge_index, edge_weight
        )
        logits, probabilities = self._edge_distribution(
            mu_c,
            mu_n,
            self._node_context(hidden, x, batch, targets),
            edge_index,
            inverse,
            pairs.size(0),
            retain_ratio,
        )
        return {
            "mu_causal": mu_c,
            "mu_non_causal": mu_n,
            "logvar_causal": logvar_c,
            "logvar_non_causal": logvar_n,
            "edge_logits": logits,
            "edge_probabilities": probabilities,
            "edge_pairs": pairs,
            "edge_inverse": inverse,
        }

    @staticmethod
    def _prune_pairs(scores, pair_weight, pair_graph, original_counts, ratio):
        active = pair_weight.detach() > 0.5
        keep = torch.zeros_like(active)
        for index in range(original_counts.numel()):
            graph_pairs = ((pair_graph == index) & active).nonzero().flatten()
            count = max(1, ceil(int(original_counts[index].item()) * ratio))
            count = min(count, graph_pairs.numel())
            if count:
                selected = scores[graph_pairs].topk(count, largest=False).indices
                keep[graph_pairs[selected]] = True
        return pair_weight * keep.float(), keep

    @staticmethod
    def _soft_prune_pairs(
        evidence, pair_weight, pair_graph, original_counts, ratio, temperature
    ):
        """Allocate the exact deletion budget as soft mass with an implicit gradient."""
        active = pair_weight.detach() > 0.5
        num_graphs = original_counts.numel()
        active_counts = torch.bincount(pair_graph[active], minlength=num_graphs)
        target_counts = torch.ceil(original_counts * ratio).long().clamp_min(1)
        target_counts = torch.minimum(target_counts, active_counts)
        remove_counts = active_counts - target_counts
        removable = active & (remove_counts[pair_graph] > 0)

        detached_scores = evidence.detach()
        lower = detached_scores.new_full((num_graphs,), torch.inf)
        upper = detached_scores.new_full((num_graphs,), -torch.inf)
        lower.scatter_reduce_(
            0,
            pair_graph[active],
            detached_scores[active],
            reduce="amin",
            include_self=True,
        )
        upper.scatter_reduce_(
            0,
            pair_graph[active],
            detached_scores[active],
            reduce="amax",
            include_self=True,
        )
        lower = lower - 20 * temperature
        upper = upper + 20 * temperature
        for _ in range(32):
            midpoint = (lower + upper) / 2
            probabilities = (
                torch.sigmoid((detached_scores - midpoint[pair_graph]) / temperature)
                * removable
            )
            mass = probabilities.new_zeros(num_graphs)
            mass.scatter_add_(0, pair_graph, probabilities)
            move_lower = mass > remove_counts
            lower = torch.where(move_lower, midpoint, lower)
            upper = torch.where(move_lower, upper, midpoint)
        detached_threshold = (lower + upper) / 2
        detached_probabilities = (
            torch.sigmoid(
                (detached_scores - detached_threshold[pair_graph]) / temperature
            )
            * removable
        )
        implicit_weights = detached_probabilities * (1 - detached_probabilities)
        center_numerator = evidence.new_zeros(num_graphs)
        center_denominator = evidence.new_zeros(num_graphs)
        center_numerator.scatter_add_(0, pair_graph, implicit_weights * evidence)
        center_denominator.scatter_add_(0, pair_graph, implicit_weights)
        weighted_center = center_numerator / center_denominator.clamp_min(1e-8)
        threshold = detached_threshold + weighted_center - weighted_center.detach()
        soft_remove = (
            torch.sigmoid((evidence - threshold[pair_graph]) / temperature) * removable
        )
        return pair_weight * (1 - soft_remove), soft_remove

    @staticmethod
    def _sample_prune_pairs(
        evidence, pair_weight, hard_keep, pair_graph, num_graphs, temperature
    ):
        active = pair_weight.detach() > 0.5
        removed = active & (~hard_keep)
        sampled_keep = active.to(pair_weight.dtype)
        graph_indices, log_probabilities = [], []
        for index in range(num_graphs):
            graph_active = ((pair_graph == index) & active).nonzero().flatten()
            graph_removed = ((pair_graph == index) & removed).nonzero().flatten()
            remove_count = graph_removed.numel()
            if remove_count <= 0 or remove_count >= graph_active.numel():
                continue
            logits = evidence[graph_active] / temperature
            sampled_positions = torch.multinomial(
                logits.detach().softmax(dim=0), remove_count, replacement=False
            )
            sampled_keep[graph_active[sampled_positions]] = 0
            graph_indices.append(index)
            log_probabilities.append(
                logits.log_softmax(dim=0)[sampled_positions].mean()
            )
        if not graph_indices:
            return sampled_keep, pair_graph.new_empty(0), evidence.new_empty(0)
        return (
            sampled_keep,
            pair_graph.new_tensor(graph_indices),
            torch.stack(log_probabilities),
        )

    @staticmethod
    def _kl_divergence(mu, logvar):
        return -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(dim=-1).mean()

    def training_objective(
        self,
        data,
        *,
        supervision_targets=None,
        temperature=2.0,
        mask_temperature=0.2,
        ranking_temperature=1.0,
        sparse_focus=2.0,
        counterfactual_samples=1,
    ):
        """Learn from the frozen task critic without edge labels or deletion targets."""
        data = data.to(self.device)
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        num_graphs = int(batch.max().item()) + 1
        pairs, inverse = undirected_edge_groups(edge_index)
        pair_graph = batch[pairs[:, 0]]
        original_counts = torch.bincount(pair_graph, minlength=num_graphs)
        pair_weight = torch.ones(pairs.size(0), device=x.device)
        with torch.no_grad():
            reference_logits = self._task_logits(
                x, edge_index, batch, edge_attr=edge_attr
            )
            reference_probabilities = F.softmax(reference_logits / temperature, dim=-1)
        targets = (
            reference_logits.argmax(dim=-1)
            if supervision_targets is None
            else torch.as_tensor(
                supervision_targets, device=x.device, dtype=torch.long
            ).view(-1)
        )

        components = {name: x.new_zeros(()) for name in self.LOSS_WEIGHTS}
        fidelity_correct = x.new_zeros(())
        initial_causal_graph = None
        for ratio in self.retain_ratios:
            stage_weight = 1 + sparse_focus * (1 - ratio) ** 2
            current_edge_weight = pair_weight[inverse]
            with torch.no_grad():
                current_logits = self._task_logits(
                    x,
                    edge_index,
                    batch,
                    edge_weight=current_edge_weight,
                    edge_attr=edge_attr,
                )
                current_probabilities = F.softmax(current_logits / temperature, dim=-1)
                current_margins = prediction_margin(current_logits, targets)
            hidden, mu_c, mu_n, logvar_c, logvar_n = self._encode_state(
                x, edge_index, current_edge_weight
            )
            pair_logits, probabilities = self._edge_distribution(
                mu_c,
                mu_n,
                self._node_context(hidden, x, batch, targets),
                edge_index,
                inverse,
                pairs.size(0),
                ratio,
            )
            evidence = pair_logits[:, 1] - pair_logits[:, 0]
            next_pair_weight, hard_keep = self._prune_pairs(
                probabilities[:, 1], pair_weight, pair_graph, original_counts, ratio
            )
            sampled_rollouts = [
                self._sample_prune_pairs(
                    evidence,
                    pair_weight,
                    hard_keep,
                    pair_graph,
                    num_graphs,
                    ranking_temperature,
                )
                for _ in range(counterfactual_samples)
            ]
            soft_pair_weight, _ = self._soft_prune_pairs(
                evidence,
                pair_weight,
                pair_graph,
                original_counts,
                ratio,
                mask_temperature,
            )
            residual_logits = self._task_logits(
                x,
                edge_index,
                batch,
                edge_weight=soft_pair_weight[inverse],
                edge_attr=edge_attr,
            )
            components["fidelity"] = components["fidelity"] + (
                stage_weight * F.cross_entropy(residual_logits, targets)
            )
            components["distillation"] = components["distillation"] + (
                stage_weight
                * F.kl_div(
                    F.log_softmax(residual_logits / temperature, dim=-1),
                    reference_probabilities,
                    reduction="batchmean",
                )
                * temperature**2
            )
            components["step_consistency"] = components["step_consistency"] + (
                stage_weight
                * F.kl_div(
                    F.log_softmax(residual_logits / temperature, dim=-1),
                    current_probabilities,
                    reduction="batchmean",
                )
                * temperature**2
            )
            residual_margins = prediction_margin(residual_logits, targets)
            components["margin"] = components["margin"] + (
                stage_weight * F.relu(current_margins - residual_margins).mean()
            )
            components["kl"] = (
                components["kl"]
                + self._kl_divergence(mu_c, logvar_c)
                + self._kl_divergence(mu_n, logvar_n)
            )
            components["independence"] = components["independence"] + (
                F.cosine_similarity(mu_c, mu_n, dim=-1).square().mean()
            )
            causal_graph = global_mean_pool(mu_c, batch)
            if initial_causal_graph is None:
                initial_causal_graph = causal_graph.detach()
            else:
                components["invariance"] = components["invariance"] + F.mse_loss(
                    causal_graph, initial_causal_graph
                )
            active_probabilities = probabilities[pair_weight.detach() > 0.5].clamp_min(
                1e-8
            )
            components["entropy"] = (
                components["entropy"]
                - (active_probabilities * active_probabilities.log()).sum(dim=-1).mean()
            )
            with torch.no_grad():
                hard_logits = self._task_logits(
                    x,
                    edge_index,
                    batch,
                    edge_weight=next_pair_weight[inverse],
                    edge_attr=edge_attr,
                )
                fidelity_correct = (
                    fidelity_correct
                    + (hard_logits.argmax(dim=-1) == targets).float().mean()
                )
                hard_margins = prediction_margin(hard_logits, targets)
                sampled_advantages = []
                for sampled_weight, sampled_graphs, _ in sampled_rollouts:
                    if not sampled_graphs.numel():
                        sampled_advantages.append(None)
                        continue
                    sampled_logits = self._task_logits(
                        x,
                        edge_index,
                        batch,
                        edge_weight=sampled_weight[inverse],
                        edge_attr=edge_attr,
                    )
                    sampled_margins = prediction_margin(sampled_logits, targets)
                    sampled_advantages.append(
                        torch.tanh(
                            sampled_margins[sampled_graphs]
                            - hard_margins[sampled_graphs]
                        ).clamp_min(0)
                    )
            policy_losses = [
                -(advantage * rollout[2]).mean()
                for rollout, advantage in zip(sampled_rollouts, sampled_advantages)
                if advantage is not None
            ]
            if policy_losses:
                components["counterfactual"] = components["counterfactual"] + (
                    stage_weight * torch.stack(policy_losses).mean()
                )
            pair_weight = next_pair_weight

        task_names = {
            "fidelity",
            "distillation",
            "margin",
            "step_consistency",
            "counterfactual",
        }
        normalizer = sum(
            1 + sparse_focus * (1 - ratio) ** 2 for ratio in self.retain_ratios
        )
        components = {
            name: value
            / (normalizer if name in task_names else len(self.retain_ratios))
            for name, value in components.items()
        }
        loss = sum(
            self.LOSS_WEIGHTS[name] * value for name, value in components.items()
        )
        return {
            "loss": loss,
            **components,
            "train_fidelity": fidelity_correct / len(self.retain_ratios),
        }

    @torch.no_grad()
    def explain_graph(self, data, retain_ratios=None, *, target=None):
        """Explain one graph; peeling makes no task-model calls for an explicit target."""
        self.eval()
        data = data.to(self.device)
        batch = data.batch
        if batch is None:
            batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
        ratios = (
            self.retain_ratios
            if retain_ratios is None
            else sorted(set(retain_ratios), reverse=True)
        )
        pairs, inverse = undirected_edge_groups(data.edge_index)
        pair_graph = torch.zeros(pairs.size(0), dtype=torch.long, device=self.device)
        original_counts = torch.tensor(
            [pairs.size(0)], dtype=torch.long, device=self.device
        )
        pair_weight = torch.ones(pairs.size(0), device=self.device)
        if target is None:
            logits = self._task_logits(
                data.x, data.edge_index, data.batch, edge_attr=data.edge_attr
            )
            target = int(logits.argmax(dim=-1).item())

        explanations = {}
        for ratio in ratios:
            hidden, mu_c, mu_n, _, _ = self._encode_state(
                data.x, data.edge_index, pair_weight[inverse]
            )
            _, probabilities = self._edge_distribution(
                mu_c,
                mu_n,
                self._node_context(
                    hidden, data.x, batch, torch.tensor([target], device=self.device)
                ),
                data.edge_index,
                inverse,
                pairs.size(0),
                ratio,
            )
            pair_weight, _ = self._prune_pairs(
                probabilities[:, 1], pair_weight, pair_graph, original_counts, ratio
            )
            mask = pair_weight[inverse].bool()
            graph = data.clone()
            graph.edge_index = data.edge_index[:, mask]
            if data.edge_attr is not None:
                graph.edge_attr = data.edge_attr[mask]
            graph.full_prediction = torch.tensor([target], device=self.device)
            graph.retained_undirected_edges = torch.tensor(
                [int(pair_weight.sum().item())], device=self.device
            )
            graph.full_undirected_edges = torch.tensor(
                [pairs.size(0)], device=self.device
            )
            explanations[format(ratio, ".10g")] = graph
        return explanations
