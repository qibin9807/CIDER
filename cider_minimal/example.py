"""Train and use CIDER on a toy dataset or your own PyG graphs."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected

from cider import GcnTaskModel, IterativeCIDER
from experiments import evaluate, fit_cider, load_graphs


def make_demo_graphs(count, generator):
    """A small feature-based classification task, only for learning the API."""
    graphs = []
    for index in range(count):
        label = index % 2
        num_nodes = int(torch.randint(8, 13, (1,), generator=generator).item())
        x = torch.randn(num_nodes, 3, generator=generator) * 0.2
        x[:, 0] += 2 * label - 1
        path = torch.stack((torch.arange(num_nodes - 1), torch.arange(1, num_nodes)))
        edge_index = to_undirected(path, num_nodes=num_nodes)
        graphs.append(Data(x=x, edge_index=edge_index, y=torch.tensor([label])))
    return graphs


def train_task_model(
    model, training, validation, *, epochs, batch_size, learning_rate, device
):
    """Select the prediction model using validation labels, before fitting CIDER."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loader = DataLoader(training, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation, batch_size=batch_size)
    best_accuracy = -1
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr
            )
            loss = F.cross_entropy(logits, batch.y.view(-1))
            loss.backward()
            optimizer.step()
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch in validation_loader:
                batch = batch.to(device)
                logits = model(
                    batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr
                )
                correct += int((logits.argmax(-1) == batch.y.view(-1)).sum().item())
        accuracy = correct / len(validation)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    print(f"Task model: best validation accuracy={best_accuracy:.2%}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, help="contains train.pt, validation.pt, test.pt"
    )
    parser.add_argument(
        "--task-checkpoint", type=Path, help="a pretrained GcnTaskModel state_dict"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("example_output"))
    parser.add_argument("--task-epochs", type=int, default=20)
    parser.add_argument("--task-lr", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=5, help="CIDER training epochs")
    parser.add_argument("--validation-interval", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare independent graph-classification splits.
    if args.data_dir is None:
        generator = torch.Generator().manual_seed(args.seed)
        training, validation, testing = [
            make_demo_graphs(count, generator) for count in (32, 8, 8)
        ]
        print("Toy data: 32 training / 8 validation / 8 test graphs (API demo only).")
    else:
        training, validation, testing = [
            load_graphs(args.data_dir / name)
            for name in ("train.pt", "validation.pt", "test.pt")
        ]

    # 2. Train your prediction model, or load an existing prediction model.
    task_config = {
        "input_dim": training[0].num_node_features,
        "label_dim": max(int(graph.y.item()) for graph in training) + 1,
    }
    task_model = GcnTaskModel(**task_config).to(device)
    if args.task_checkpoint is None:
        train_task_model(
            task_model,
            training,
            validation,
            epochs=args.task_epochs,
            batch_size=args.batch_size,
            learning_rate=args.task_lr,
            device=device,
        )
    else:
        task_model.load_state_dict(
            torch.load(args.task_checkpoint, map_location=device, weights_only=True)
        )
    torch.save(task_model.state_dict(), args.output_dir / "task_model.pt")

    # 3. CIDER freezes this classifier and learns from its prediction feedback.
    cider_config = {"hidden_channels": 48, "num_classes": task_config["label_dim"]}
    model = IterativeCIDER(task_config["input_dim"], task_model, **cider_config).to(
        device
    )
    checkpoint_path = args.output_dir / "model.pt"
    fit_cider(
        model,
        training,
        validation,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        validation_interval=args.validation_interval,
        checkpoint=checkpoint_path,
    )

    # Record the constructors beside the selected weights for later reuse.
    selected = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    selected.update(task_config=task_config, cider_config=cider_config)
    torch.save(selected, checkpoint_path)

    # 4. Rebuild a model from the file, then evaluate the held-out test split.
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    task_model = GcnTaskModel(**saved["task_config"])
    restored = IterativeCIDER(
        saved["task_config"]["input_dim"], task_model, **saved["cider_config"]
    ).to(device)
    restored.load_state_dict(saved["model_state_dict"])
    restored.eval()
    metrics = evaluate(restored, testing, batch_size=args.batch_size)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    # 5. Explain one new graph. Pick the target once; all nine deletion rounds
    # then use CIDER alone. Keep the default ratios to preserve its full rollout.
    graph = testing[0].clone().to(device)
    with torch.no_grad():
        target = int(
            restored.task_model(graph.x, graph.edge_index, edge_attr=graph.edge_attr)
            .argmax(-1)
            .item()
        )
    explanations = restored.explain_graph(graph, target=target)
    torch.save(
        {ratio: residual.cpu() for ratio, residual in explanations.items()},
        args.output_dir / "explanations.pt",
    )
    torch.save(testing[0].clone().cpu(), args.output_dir / "example_graph.pt")
    print(f"Selected CIDER epoch: {saved['best_epoch']}")
    print(f"Test full-graph label accuracy: {metrics['full']['accuracy']:.2%}")
    print(
        f"10% subgraph: fidelity={metrics['0.1']['fidelity']:.2%}, "
        f"label accuracy={metrics['0.1']['accuracy']:.2%}"
    )
    print(
        f"Saved model, metrics, and nine residual graphs to {args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
