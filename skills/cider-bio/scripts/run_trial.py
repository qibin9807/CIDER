"""Train one validation-selected expression-graph CIDER trial."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.loader import DataLoader

from bio_modeling import (
    IterativeCIDER,
    build_cider,
    build_task,
    classification_metrics,
    check_run_inputs,
    evaluate_cider,
    evaluate_task,
    environment_info,
    load_inputs,
    seed_everything,
    selection_summary,
)


DEFAULTS = {
    "seed": 0,
    "batch_size": 16,
    "device": "cpu",
    "torch_threads": 1,
    "task": {
        "hidden_dim": 32,
        "embedding_dim": 16,
        "num_layers": 2,
        "pred_hidden_dims": [16],
        "learning_rate": 0.001,
        "epochs": 20,
    },
    "cider": {
        "hidden_channels": 32,
        "latent_channels": 16,
        "encoder_layers": 2,
        "decoder_channels": 32,
        "dropout": 0.0,
        "learning_rate": 0.001,
        "epochs": 10,
        "validation_interval": 5,
        "temperature": 2.0,
        "mask_temperature": 0.2,
        "ranking_temperature": 1.0,
        "sparse_focus": 2.0,
        "counterfactual_samples": 1,
        "loss_weights": dict(IterativeCIDER.LOSS_WEIGHTS),
    },
}


def resolved_config(requested):
    config = deepcopy(DEFAULTS)
    for key, value in requested.items():
        if key in ("task", "cider"):
            config[key].update(value)
        else:
            config[key] = value
    config["cider"]["loss_weights"] = {
        **IterativeCIDER.LOSS_WEIGHTS,
        **requested.get("cider", {}).get("loss_weights", {}),
    }
    return config


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def train_task(model, training, validation, config, batch_size, num_classes, seed):
    loader = DataLoader(
        training,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"], weight_decay=1e-4
    )
    history, best_key = [], None
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        for batch in loader:
            batch = batch.to(next(model.parameters()).device)
            optimizer.zero_grad()
            loss = F.cross_entropy(
                model(batch.x, batch.edge_index, batch.batch), batch.y.view(-1)
            )
            loss.backward()
            optimizer.step()
        metrics = evaluate_task(model, validation, batch_size, num_classes)
        history.append({"epoch": epoch, "validation": metrics})
        key = (metrics["macro_f1"], metrics["accuracy"], -metrics["loss"])
        if best_key is None or key > best_key:
            best_key = key
            best_state, best_metrics, best_epoch = (
                deepcopy(model.state_dict()),
                metrics,
                epoch,
            )
        print(json.dumps({"stage": "task", **history[-1]}), flush=True)
    model.load_state_dict(best_state)
    return best_metrics, best_epoch, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-run", type=Path)
    parser.add_argument(
        "--task-only",
        action="store_true",
        help="select the task classifier on validation data without training CIDER",
    )
    args = parser.parse_args()
    requested = json.loads(args.config.read_text(encoding="utf-8"))
    config = resolved_config(requested)
    torch.set_num_threads(config["torch_threads"])
    seed_everything(config["seed"])
    datasets, metadata = load_inputs(
        args.prepared_dir, args.network, max_genes=config.get("max_genes")
    )
    context = {
        key: metadata[key]
        for key in ("genes", "classes", "corpus_sha256", "network_sha256")
    }
    num_genes, num_classes = len(metadata["genes"]), len(metadata["classes"])
    history = {"task": [], "cider": []}
    if args.task_run is not None:
        task_checkpoint = torch.load(
            args.task_run / "task.pt", map_location="cpu", weights_only=False
        )
        check_run_inputs(task_checkpoint, metadata)
        for key, value in requested.get("task", {}).items():
            if value != task_checkpoint["task_config"][key]:
                raise ValueError(
                    f"--task-run reuses its task configuration; {key} differs."
                )
        config["task"] = task_checkpoint["task_config"]
    if config["task"]["epochs"] < 1:
        raise ValueError("Task epochs must be positive.")
    if (
        not args.task_only
        and min(config["cider"]["epochs"], config["cider"]["validation_interval"]) < 1
    ):
        raise ValueError("CIDER epochs and validation_interval must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "config.json", config)
    save_json(args.output_dir / "environment.json", environment_info(config["device"]))
    majority_class = int(
        np.bincount(datasets["train"].labels, minlength=num_classes).argmax()
    )
    majority_baseline = {
        "fit_split": "train",
        "predicted_class": metadata["classes"][majority_class],
        "predicted_class_index": majority_class,
        "validation": classification_metrics(
            datasets["validation"].labels,
            np.full(len(datasets["validation"]), majority_class, dtype=np.int64),
            num_classes,
        ),
    }
    task_model = build_task(config["task"], num_genes, num_classes).to(config["device"])
    if args.task_run is None:
        task_metrics, task_epoch, history["task"] = train_task(
            task_model,
            datasets["train"],
            datasets["validation"],
            config["task"],
            config["batch_size"],
            num_classes,
            config["seed"],
        )
        task_checkpoint = {
            **context,
            "task_config": config["task"],
            "seed": config["seed"],
            "model_state_dict": deepcopy(task_model.state_dict()),
            "validation": task_metrics,
            "best_epoch": task_epoch,
            "selection_metric": "validation_macro_f1",
        }
    else:
        task_model.load_state_dict(task_checkpoint["model_state_dict"])
        task_metrics, task_epoch = (
            task_checkpoint["validation"],
            task_checkpoint["best_epoch"],
        )
    torch.save(task_checkpoint, args.output_dir / "task.pt")
    if args.task_only:
        save_json(args.output_dir / "history.json", {"task": history["task"]})
        save_json(
            args.output_dir / "metrics.json",
            {
                **context,
                "seed": config["seed"],
                "task_only": True,
                "selection_split": "validation",
                "task_validation": task_metrics,
                "best_task_epoch": task_epoch,
                "majority_baseline": majority_baseline,
                "task_run": str(args.task_run) if args.task_run is not None else None,
            },
        )
        return

    # A fixed task model and policy seed make CIDER-only trials comparable.
    seed_everything(config["seed"])
    model = build_cider(task_model, config["cider"], num_genes, num_classes).to(
        config["device"]
    )
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.Adam(
        parameters, lr=config["cider"]["learning_rate"], weight_decay=1e-5
    )
    loader = DataLoader(
        datasets["train"],
        batch_size=config["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(config["seed"]),
    )
    objective_config = {
        key: config["cider"][key]
        for key in (
            "temperature",
            "mask_temperature",
            "ranking_temperature",
            "sparse_focus",
            "counterfactual_samples",
        )
        if key in config["cider"]
    }
    best_score = float("-inf")
    for epoch in range(1, config["cider"]["epochs"] + 1):
        model.train()
        totals, count = {}, 0
        for batch in loader:
            optimizer.zero_grad()
            output = model.training_objective(batch, **objective_config)
            output["loss"].backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            count += batch.num_graphs
            for key, value in output.items():
                totals[key] = (
                    totals.get(key, 0.0) + float(value.detach()) * batch.num_graphs
                )
        record = {
            "epoch": epoch,
            "train": {key: value / count for key, value in totals.items()},
        }
        if (
            epoch % config["cider"]["validation_interval"] == 0
            or epoch == config["cider"]["epochs"]
        ):
            metrics = evaluate_cider(
                model, datasets["validation"], config["batch_size"], num_classes
            )
            summary = selection_summary(metrics)
            record.update(validation=metrics, selection=summary)
            if summary["selection_score"] > best_score:
                best_score, best_epoch, best_metrics = (
                    summary["selection_score"],
                    epoch,
                    metrics,
                )
                torch.save(
                    {
                        **context,
                        "task_config": config["task"],
                        "cider_config": config["cider"],
                        "seed": config["seed"],
                        "task_seed": task_checkpoint["seed"],
                        "model_state_dict": model.state_dict(),
                        "best_epoch": epoch,
                        "validation": metrics,
                        "selection": summary,
                        "task_validation": task_metrics,
                        "teacher_free": True,
                    },
                    args.output_dir / "selected.pt",
                )
        history["cider"].append(record)
        print(json.dumps({"stage": "cider", **record}), flush=True)
        save_json(args.output_dir / "history.json", history)
    save_json(
        args.output_dir / "metrics.json",
        {
            **context,
            "seed": config["seed"],
            "selection_split": "validation",
            "task_validation": task_metrics,
            "best_task_epoch": task_epoch,
            "cider_validation": best_metrics,
            "best_cider_epoch": best_epoch,
            "selection": selection_summary(best_metrics),
            "task_run": str(args.task_run) if args.task_run is not None else None,
            "teacher_free": True,
            "majority_baseline": majority_baseline,
        },
    )


if __name__ == "__main__":
    main()
