"""Checkpoint loading, validation-selected training, and experiment evaluation."""

from copy import deepcopy
from pathlib import Path

import networkx as nx
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader

from ba2 import BA2MotifModel
from cider import (
    DEFAULT_RETAIN_RATIOS,
    GcnTaskModel,
    IterativeCIDER,
    undirected_edge_groups,
)


def load_graphs(path):
    """Read a list of PyG Data graphs or the repository's (data, slices) files."""
    graphs = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(graphs, list):
        return graphs
    dataset = InMemoryDataset()
    dataset.data, dataset.slices = graphs
    return [dataset[index] for index in range(len(dataset))]


def load_best_model(experiment, device="cpu"):
    directory = Path(__file__).parent / "params" / experiment
    checkpoint = torch.load(
        directory / "best.pt", map_location=device, weights_only=False
    )
    if experiment == "ba2":
        model = BA2MotifModel()
        model.load_state_dict(
            {
                **checkpoint["graph_model_state_dict"],
                **checkpoint["edge_model_state_dict"],
            }
        )
    else:
        task = torch.load(
            directory / "task_model.pt", map_location=device, weights_only=False
        )
        input_dim = {"nci1": 37, "mutag": 14}[experiment]
        classifier = GcnTaskModel(input_dim)
        config = checkpoint["config"]
        model = IterativeCIDER(
            input_dim,
            classifier,
            hidden_channels=config["hidden_channels"],
            latent_channels=config["latent_channels"],
            encoder_layers=config["encoder_layers"],
            decoder_channels=config["decoder_channels"],
            dropout=config["dropout"],
        )
        state = {
            "task_model." + name: tensor
            for name, tensor in task["model_state_dict"].items()
        }
        state.update(checkpoint["model_state_dict"])
        model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def evaluate(model, graphs, batch_size=128):
    """Evaluate all nine recursive stages, with task calls outside edge selection."""
    model.eval()
    ratios = [format(value, ".10g") for value in DEFAULT_RETAIN_RATIOS]
    residuals = {ratio: [] for ratio in ratios}
    full_correct = 0
    for data in graphs:
        explanations = model.explain_graph(data.clone())
        full_prediction = int(explanations[ratios[0]].full_prediction.item())
        full_correct += int(full_prediction == int(data.y.view(-1)[0].item()))
        for ratio, graph in explanations.items():
            residuals[ratio].append(graph.cpu())

    count = len(graphs)
    metrics = {
        "full": {
            "fidelity": 1.0,
            "accuracy": full_correct / count,
            "retained_ratio": 1.0,
        }
    }
    for ratio in ratios:
        fidelity = accuracy = 0
        retained = 0.0
        for batch in DataLoader(residuals[ratio], batch_size=batch_size, shuffle=False):
            batch = batch.to(model.device)
            predictions = model._task_logits(
                batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr
            ).argmax(dim=1)
            fidelity += int(
                (predictions == batch.full_prediction.view(-1)).sum().item()
            )
            accuracy += int((predictions == batch.y.view(-1)).sum().item())
            retained += float(
                (
                    batch.retained_undirected_edges.float()
                    / batch.full_undirected_edges.float()
                )
                .sum()
                .item()
            )
        metrics[ratio] = {
            "fidelity": fidelity / count,
            "accuracy": accuracy / count,
            "retained_ratio": retained / count,
            "pre_refinement_fidelity": fidelity / count,
        }
    return metrics


def _selection_score(metrics):
    fidelity = [
        metrics[format(value, ".10g")]["fidelity"] for value in DEFAULT_RETAIN_RATIOS
    ]
    mean = sum(fidelity) / 9
    low_ratio_mean = sum(fidelity[:3]) / 3
    return 0.7 * mean + 0.3 * low_ratio_mean


def fit_cider(
    model,
    train_graphs,
    validation_graphs,
    *,
    epochs=30,
    batch_size=128,
    learning_rate=1e-3,
    seed=0,
    validation_interval=5,
    checkpoint=None,
):
    """Train without edge labels and restore the best validation checkpoint."""
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_graphs,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=model.device.type == "cuda",
    )
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=1e-5)
    best_score = float("-inf")
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        totals = {}
        count = 0
        for batch in loader:
            optimizer.zero_grad()
            output = model.training_objective(batch)
            output["loss"].backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            count += batch.num_graphs
            for name, value in output.items():
                totals[name] = (
                    totals.get(name, 0.0) + float(value.detach()) * batch.num_graphs
                )
        record = {
            "epoch": epoch,
            "train": {name: value / count for name, value in totals.items()},
        }
        if epoch % validation_interval == 0 or epoch == epochs:
            metrics = evaluate(model, validation_graphs, batch_size)
            score = _selection_score(metrics)
            record.update(validation=metrics, validation_score=score)
            if score > best_score:
                best_score = score
                best_state = deepcopy(model.state_dict())
                if checkpoint is not None:
                    torch.save(
                        {
                            "model_state_dict": best_state,
                            "best_epoch": epoch,
                            "validation": metrics,
                            "validation_score": score,
                        },
                        checkpoint,
                    )
        history.append(record)
        print(f"epoch {epoch}: loss={record['train']['loss']:.6f}")
    model.load_state_dict(best_state)
    return history


def _attributed_graph(data):
    graph = nx.Graph()
    for index in range(data.num_nodes):
        graph.add_node(index, features=repr(tuple(data.x[index].reshape(-1).tolist())))
    for position, (u, v) in enumerate(data.edge_index.t().tolist()):
        if not graph.has_edge(u, v):
            features = (
                ""
                if data.edge_attr is None
                else repr(tuple(data.edge_attr[position].reshape(-1).tolist()))
            )
            graph.add_edge(u, v, features=features)
    return graph


def _clean_splits(training, validation, test):
    """Exclude attributed graph isomorphisms, without consulting graph labels."""
    node_match = nx.algorithms.isomorphism.categorical_node_match("features", "")
    edge_match = nx.algorithms.isomorphism.categorical_edge_match("features", "")

    def add_if_new(data, prior_indices, destination):
        graph = _attributed_graph(data)
        key = nx.weisfeiler_lehman_graph_hash(
            graph, node_attr="features", edge_attr="features", iterations=8
        )
        for index in prior_indices:
            for other in index.get(key, []):
                if nx.is_isomorphic(
                    graph, other, node_match=node_match, edge_match=edge_match
                ):
                    return False
        destination.setdefault(key, []).append(graph)
        return True

    training_index, validation_index, test_index = {}, {}, {}
    for data in training:
        add_if_new(data, [], training_index)
    clean_validation = [
        data
        for data in validation
        if add_if_new(data, [training_index, validation_index], validation_index)
    ]
    clean_test = [
        data
        for data in test
        if add_if_new(data, [training_index, validation_index, test_index], test_index)
    ]
    return clean_validation, clean_test


@torch.no_grad()
def _evaluate_ba2(model, graphs):
    ratios = [format(value, ".10g") for value in DEFAULT_RETAIN_RATIOS]
    correct = {ratio: 0 for ratio in ratios}
    recall = {ratio: 0.0 for ratio in ratios}
    retained = {ratio: 0.0 for ratio in ratios}
    full_correct = pruned_correct = true_positive = false_positive = false_negative = (
        true_negative
    ) = 0
    valid_fraction = 0.0
    device = next(model.parameters()).device
    for data in graphs:
        data = data.clone().to(device)
        source, target = data.edge_index
        valid_target = (source >= 20) & (target >= 20) & (source != target)
        valid_logits, _ = model.edge_logits(data.edge_index, data.num_nodes)
        valid = valid_logits >= 0
        true_positive += int((valid & valid_target).sum())
        false_positive += int((valid & ~valid_target).sum())
        false_negative += int((~valid & valid_target).sum())
        true_negative += int((~valid & ~valid_target).sum())
        full_correct += int(
            model(data.x, data.edge_index).argmax(-1).item() == data.y.item()
        )
        pruned_correct += int(
            model(data.x, data.edge_index[:, valid]).argmax(-1).item() == data.y.item()
        )
        valid_fraction += float(valid.float().mean())
        target_edges = set(map(tuple, data.edge_index[:, valid_target].t().tolist()))
        pair_count = len(undirected_edge_groups(data.edge_index)[0])
        for ratio, graph in model.explain_graph(data).items():
            prediction = model(graph.x, graph.edge_index).argmax(-1).item()
            correct[ratio] += int(prediction == data.y.item())
            selected_edges = set(map(tuple, graph.edge_index.t().tolist()))
            recall[ratio] += len(selected_edges & target_edges) / len(target_edges)
            retained[ratio] += (
                len(undirected_edge_groups(graph.edge_index)[0]) / pair_count
            )
    count = len(graphs)
    precision = true_positive / max(true_positive + false_positive, 1)
    edge_recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "full_graph_accuracy": full_correct / count,
        "invalid_edges_removed_accuracy": pruned_correct / count,
        "mean_retained_directed_edge_fraction": valid_fraction / count,
        "edge_discriminator": {
            "accuracy": (true_positive + true_negative)
            / (true_positive + false_positive + false_negative + true_negative),
            "precision": precision,
            "recall": edge_recall,
            "f1": 2 * precision * edge_recall / max(precision + edge_recall, 1e-12),
        },
        "cider_retention": {
            ratio: {
                "accuracy": correct[ratio] / count,
                "valid_edge_recall": recall[ratio] / count,
                "actual_retained_pair_fraction": retained[ratio] / count,
            }
            for ratio in ratios
        },
    }


def reproduce(
    experiment, data_root, split, device="cpu", batch_size=128, torch_threads=8
):
    """Recompute a reported split from the original processed data files."""
    torch.set_num_threads(torch_threads)
    model = load_best_model(experiment, device)
    dataset = {"nci1": "NCI1", "mutag": "MUTAG", "ba2": "BA-2motif"}[experiment]
    directory = Path(data_root) / dataset / "processed"
    validation = load_graphs(directory / "evaluation.pt")
    test = load_graphs(directory / "testing.pt")
    if experiment == "ba2":
        return _evaluate_ba2(model, validation if split == "validation" else test)
    training = load_graphs(directory / "training.pt")
    validation, test = _clean_splits(training, validation, test)
    return evaluate(model, validation if split == "validation" else test, batch_size)
