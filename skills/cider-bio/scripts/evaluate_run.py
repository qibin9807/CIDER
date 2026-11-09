"""Evaluate one locked CIDER run and export selected gene edges."""

import argparse
import csv
import json
from pathlib import Path

import torch

from bio_modeling import (
    build_cider,
    build_task,
    check_run_inputs,
    evaluate_cider,
    evaluate_task,
    environment_info,
    load_inputs,
    selection_summary,
)
from cider import undirected_edge_groups


@torch.no_grad()
def export_edges(model, dataset, metadata, split, path, max_explanations):
    correlations = {
        tuple(sorted(pair)): float(value)
        for pair, value in zip(metadata["pairs"].tolist(), metadata["correlation"])
    }
    fields = [
        "sample_id",
        "split",
        "label",
        "target_class",
        "retain_ratio",
        "gene_source",
        "gene_target",
        "correlation",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(min(max_explanations, len(dataset))):
            graph = dataset[index]
            label = metadata["classes"][int(graph.y.item())]
            for ratio, explanation in model.explain_graph(graph).items():
                target = metadata["classes"][int(explanation.full_prediction.item())]
                pairs = undirected_edge_groups(explanation.edge_index)[0].cpu().tolist()
                for source, target_gene in pairs:
                    writer.writerow(
                        {
                            "sample_id": dataset.sample_ids[index],
                            "split": split,
                            "label": label,
                            "target_class": target,
                            "retain_ratio": ratio,
                            "gene_source": metadata["genes"][source],
                            "gene_target": metadata["genes"][target_gene],
                            "correlation": correlations[(source, target_gene)],
                        }
                    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-explanations", type=int, default=5)
    parser.add_argument("--device")
    args = parser.parse_args()
    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    torch.set_num_threads(config["torch_threads"])
    checkpoint = torch.load(
        args.run_dir / "selected.pt", map_location="cpu", weights_only=False
    )
    datasets, metadata = load_inputs(
        args.prepared_dir,
        args.network,
        splits=(args.split,),
        max_genes=config.get("max_genes"),
    )
    check_run_inputs(checkpoint, metadata)
    num_genes, num_classes = len(metadata["genes"]), len(metadata["classes"])
    task_model = build_task(checkpoint["task_config"], num_genes, num_classes)
    model = build_cider(task_model, checkpoint["cider_config"], num_genes, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(args.device or config["device"]).eval()
    dataset = datasets[args.split]
    metrics = evaluate_cider(model, dataset, config["batch_size"], num_classes)
    report = {
        "split": args.split,
        "selection_split": "validation",
        "samples": len(dataset),
        "corpus_sha256": metadata["corpus_sha256"],
        "network_sha256": metadata["network_sha256"],
        "task": evaluate_task(
            model.task_model, dataset, config["batch_size"], num_classes
        ),
        "cider": metrics,
        "summary": selection_summary(metrics),
        "best_epoch": checkpoint["best_epoch"],
        "classes": metadata["classes"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "environment.json").write_text(
        json.dumps(environment_info(model.device), indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    export_edges(
        model,
        dataset,
        metadata,
        args.split,
        args.output_dir / "selected_edges.csv",
        args.max_explanations,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
