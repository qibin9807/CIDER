"""Show the selected CIDER results, or recompute them from graph data."""

import argparse
import json
from pathlib import Path


EXPERIMENTS = ("nci1", "mutag", "ba2")


def percent(value):
    return f"{100 * value:6.2f}%"


def show_result(info, split, metrics):
    print(f"\n{info['title']} — {split} ({info[split]['num_graphs']} graphs)")
    print(f"Model: {info['model']}")
    print(f"Selected seed/epoch: {info['seed']}/{info['best_epoch']}")
    print(f"Checkpoint: {info['checkpoint']}")
    print("Config: " + ", ".join(f"{k}={v}" for k, v in info["config"].items()))

    if "cider_retention" in metrics:
        edges = metrics["edge_discriminator"]
        print(f"Full-graph accuracy: {percent(metrics['full_graph_accuracy'])}")
        print(f"Pruned accuracy: {percent(metrics['invalid_edges_removed_accuracy'])}")
        print(
            "Edge precision/recall/F1: "
            + " / ".join(percent(edges[key]) for key in ("precision", "recall", "f1"))
        )
        print(
            "Valid-edge fraction: "
            + percent(metrics["mean_retained_directed_edge_fraction"])
        )
        print(" Retained    Actual  Accuracy  Valid recall")
        for ratio, row in metrics["cider_retention"].items():
            print(
                f" {percent(float(ratio))}  {percent(row['actual_retained_pair_fraction'])}"
                f"  {percent(row['accuracy'])}  {percent(row['valid_edge_recall'])}"
            )
    else:
        ratios = [str(value / 10) for value in range(1, 10)]
        mean_fidelity = sum(metrics[r]["fidelity"] for r in ratios) / 9
        low_fidelity = sum(metrics[r]["fidelity"] for r in ratios[:3]) / 3
        mean_accuracy = sum(metrics[r]["accuracy"] for r in ratios) / 9
        print(f"Full-graph accuracy: {percent(metrics['full']['accuracy'])}")
        print(f"Mean fidelity: {percent(mean_fidelity)}")
        print(f"10%-30% fidelity: {percent(low_fidelity)}")
        print(f"Mean label accuracy: {percent(mean_accuracy)}")
        print(" Retained    Actual  Fidelity  Accuracy")
        for ratio in ratios:
            row = metrics[ratio]
            print(
                f" {percent(float(ratio))}  {percent(row['retained_ratio'])}"
                f"  {percent(row['fidelity'])}  {percent(row['accuracy'])}"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-e",
        "--experiment",
        nargs="+",
        choices=(*EXPERIMENTS, "all"),
        default=["all"],
        help="experiments to display; default: all three",
    )
    parser.add_argument(
        "--split", choices=("validation", "test", "both"), default="test"
    )
    parser.add_argument(
        "--recompute", action="store_true", help="evaluate the bundled weights"
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path(__file__).parent.parent / "data"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--torch-threads", type=int, default=8)
    args = parser.parse_args()
    results = json.loads((Path(__file__).parent / "results.json").read_text())
    selected = (
        EXPERIMENTS if "all" in args.experiment else dict.fromkeys(args.experiment)
    )
    splits = ("validation", "test") if args.split == "both" else (args.split,)

    if args.recompute:
        from experiments import reproduce

    print(
        "Results: "
        + (
            "recomputed from graph data"
            if args.recompute
            else "archived in results.json"
        )
    )
    for name in selected:
        for split in splits:
            metrics = (
                reproduce(
                    name,
                    args.data_root,
                    split,
                    args.device,
                    batch_size=args.batch_size,
                    torch_threads=args.torch_threads,
                )
                if args.recompute
                else results[name][split]["metrics"]
            )
            show_result(results[name], split, metrics)


if __name__ == "__main__":
    main()
