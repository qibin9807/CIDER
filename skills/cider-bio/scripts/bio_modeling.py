"""Expression graphs and metrics shared by the CIDER biology scripts."""

from pathlib import Path
import hashlib
from importlib.metadata import version
import platform
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).parents[1] / "assets" / "cider"))
from cider import DEFAULT_RETAIN_RATIOS, GcnTaskModel, IterativeCIDER


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def environment_info(device):
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_geometric": version("torch-geometric"),
        "numpy": np.__version__,
        "scikit_learn": version("scikit-learn"),
        "device": str(device),
        "torch_threads": torch.get_num_threads(),
        "core_sha256": sha256(
            Path(__file__).parents[1] / "assets" / "cider" / "cider.py"
        ),
    }


class ExpressionGraphs(Dataset):
    """One graph per sample, with expression and a fixed gene-identity feature."""

    def __init__(self, expression, labels, sample_ids, edge_index, mean, std):
        self.expression = expression
        self.labels = labels
        self.sample_ids = sample_ids
        self.edge_index = torch.as_tensor(edge_index, dtype=torch.long)
        self.mean, self.std = mean, std
        self.identity = torch.eye(expression.shape[1])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        expression = torch.from_numpy(
            ((self.expression[index] - self.mean) / self.std).astype(np.float32)
        ).unsqueeze(-1)
        return Data(
            x=torch.cat((expression, self.identity), dim=-1),
            edge_index=self.edge_index,
            y=torch.tensor([int(self.labels[index])], dtype=torch.long),
        )


def load_inputs(
    prepared_dir, network_path, splits=("train", "validation"), max_genes=None
):
    """Check network provenance and materialize only the requested sample rows."""
    prepared_path = Path(prepared_dir) / "prepared.npz"
    corpus_hash = sha256(prepared_path)
    with (
        np.load(prepared_path, allow_pickle=False) as corpus,
        np.load(network_path, allow_pickle=False) as network,
    ):
        genes, classes = corpus["genes"].astype(str), corpus["classes"].astype(str)
        if str(network["data_sha256"].item()) != corpus_hash:
            raise ValueError("The network was not built from this prepared.npz.")
        if not np.array_equal(network["genes"].astype(str), genes):
            raise ValueError(
                "Network genes and prepared genes must have the same order."
            )
        if not np.array_equal(network["train_indices"], corpus["train_indices"]):
            raise ValueError(
                "The network must use exactly the prepared training indices."
            )
        if max_genes is not None and len(genes) > max_genes:
            raise ValueError(
                f"Found {len(genes)} genes; configured max_genes={max_genes}."
            )
        edges = network["edge_index"].astype(np.int64)
        if edges.shape[1] == 0:
            raise ValueError("CIDER requires a network with at least one edge.")
        mean, std = corpus["mean"].astype(np.float32), corpus["std"].astype(np.float32)
        if np.any(std <= 0):
            raise ValueError("Prepared training standard deviations must be positive.")
        indices = {split: corpus[f"{split}_indices"] for split in splits}
        # NPZ decompresses an array as a whole; only selected rows become graph data.
        expression, labels, sample_ids = (
            corpus["expression"],
            corpus["labels"],
            corpus["sample_ids"],
        )
        datasets = {
            split: ExpressionGraphs(
                expression[rows].astype(np.float32),
                labels[rows].astype(np.int64),
                sample_ids[rows].astype(str),
                edges,
                mean,
                std,
            )
            for split, rows in indices.items()
        }
        metadata = {
            "genes": genes.tolist(),
            "classes": classes.tolist(),
            "corpus_sha256": corpus_hash,
            "network_sha256": sha256(network_path),
            "pairs": network["pairs"].astype(np.int64),
            "correlation": network["correlation"].astype(float),
        }
    return datasets, metadata


def build_task(task_config, num_genes, num_classes):
    architecture = {
        key: task_config[key]
        for key in ("hidden_dim", "embedding_dim", "num_layers", "pred_hidden_dims")
    }
    return GcnTaskModel(num_genes + 1, label_dim=num_classes, **architecture)


def build_cider(task_model, cider_config, num_genes, num_classes):
    architecture = {
        key: cider_config[key]
        for key in (
            "hidden_channels",
            "latent_channels",
            "encoder_layers",
            "decoder_channels",
            "dropout",
        )
    }
    model = IterativeCIDER(
        num_genes + 1, task_model, num_classes=num_classes, **architecture
    )
    weights = cider_config.get("loss_weights", {})
    if set(weights) - set(model.LOSS_WEIGHTS):
        raise ValueError(
            "loss_weights may only change the teacher-free CIDER loss terms."
        )
    model.LOSS_WEIGHTS = {**model.LOSS_WEIGHTS, **weights}
    return model


def classification_metrics(labels, predictions, num_classes):
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (np.asarray(labels), np.asarray(predictions)), 1)
    support, predicted = confusion.sum(axis=1), confusion.sum(axis=0)
    true_positive = confusion.diagonal()
    recall = np.divide(
        true_positive, support, out=np.zeros(num_classes), where=support > 0
    )
    f1 = np.divide(
        2 * true_positive,
        support + predicted,
        out=np.zeros(num_classes),
        where=support + predicted > 0,
    )
    return {
        "accuracy": float(true_positive.sum() / confusion.sum()),
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall[support > 0].mean()),
        "confusion_matrix": confusion.tolist(),
        "per_class": [
            {
                "class_index": index,
                "support": int(support[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
            for index in range(num_classes)
        ],
    }


@torch.no_grad()
def evaluate_task(model, dataset, batch_size, num_classes):
    model.eval()
    device = next(model.parameters()).device
    truth, predictions, loss = [], [], 0.0
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        loss += float(F.cross_entropy(logits, batch.y.view(-1), reduction="sum"))
        truth.extend(batch.y.view(-1).cpu().tolist())
        predictions.extend(logits.argmax(-1).cpu().tolist())
    return {
        **classification_metrics(truth, predictions, num_classes),
        "loss": loss / len(dataset),
    }


@torch.no_grad()
def evaluate_cider(model, dataset, batch_size, num_classes):
    """Stream one batch of residuals while keeping the nine-stage CIDER policy intact."""
    model.eval()
    ratios = [format(ratio, ".10g") for ratio in DEFAULT_RETAIN_RATIOS]
    predictions = {ratio: [] for ratio in ratios}
    fidelity = {ratio: 0 for ratio in ratios}
    retained = {ratio: 0.0 for ratio in ratios}
    truth, full_predictions = [], []
    for start in range(0, len(dataset), batch_size):
        residuals = {ratio: [] for ratio in ratios}
        for index in range(start, min(start + batch_size, len(dataset))):
            graph = dataset[index]
            explanations = model.explain_graph(graph)
            full_predictions.append(int(explanations[ratios[0]].full_prediction.item()))
            truth.append(int(graph.y.item()))
            for ratio, explanation in explanations.items():
                residuals[ratio].append(explanation.cpu())
        for ratio in ratios:
            batch = Batch.from_data_list(residuals[ratio]).to(model.device)
            selected = model.task_model(batch.x, batch.edge_index, batch.batch).argmax(
                -1
            )
            predictions[ratio].extend(selected.cpu().tolist())
            fidelity[ratio] += int((selected == batch.full_prediction.view(-1)).sum())
            retained[ratio] += float(
                (
                    batch.retained_undirected_edges.float()
                    / batch.full_undirected_edges.float()
                ).sum()
            )
    metrics = {
        "full": {
            **classification_metrics(truth, full_predictions, num_classes),
            "fidelity": 1.0,
            "retained_ratio": 1.0,
        }
    }
    for ratio in ratios:
        metrics[ratio] = {
            **classification_metrics(truth, predictions[ratio], num_classes),
            "fidelity": fidelity[ratio] / len(dataset),
            "retained_ratio": retained[ratio] / len(dataset),
        }
    return metrics


def selection_summary(metrics):
    ratios = [format(ratio, ".10g") for ratio in DEFAULT_RETAIN_RATIOS]
    mean_fidelity = sum(metrics[ratio]["fidelity"] for ratio in ratios) / 9
    low_fidelity = sum(metrics[ratio]["fidelity"] for ratio in ratios[:3]) / 3
    return {
        "mean_fidelity": mean_fidelity,
        "low_ratio_fidelity": low_fidelity,
        "selection_score": 0.7 * mean_fidelity + 0.3 * low_fidelity,
        "mean_accuracy": sum(metrics[ratio]["accuracy"] for ratio in ratios) / 9,
    }


def check_run_inputs(checkpoint, metadata):
    for key in ("corpus_sha256", "network_sha256", "genes", "classes"):
        if checkpoint[key] != metadata[key]:
            raise ValueError(f"The saved run and current inputs differ in {key}.")
