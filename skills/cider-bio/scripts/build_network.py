"""Build an undirected candidate gene network from training-only correlations."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from bio_io import load_bio_data


def correlation_edges(expression, *, method, min_correlation, top_k, block_size=256):
    """Keep the union of each gene's strongest qualifying neighbors."""
    values = np.asarray(expression, dtype=np.float64)
    if values.shape[0] < 3 or values.shape[1] < 2:
        raise ValueError(
            "Correlations require at least three training samples and two genes."
        )
    if not np.isfinite(values).all():
        raise ValueError("The training expression matrix contains non-finite values.")
    if not 0 <= min_correlation <= 1 or top_k < 1:
        raise ValueError("Use a correlation threshold in [0, 1] and a positive top-k.")
    if method == "spearman":
        values = rankdata(values, axis=0, method="average")
    values = values - values.mean(axis=0)
    norms = np.sqrt((values * values).sum(axis=0))
    values = np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)

    selected = set()
    gene_count = values.shape[1]
    # Blocks avoid allocating a full G x G correlation matrix for large gene sets.
    for start in range(0, gene_count, block_size):
        correlations = values[:, start : start + block_size].T @ values
        for offset, row in enumerate(correlations):
            gene = start + offset
            magnitude = np.abs(row)
            neighbors = np.flatnonzero((magnitude >= min_correlation) & (magnitude > 0))
            neighbors = neighbors[neighbors != gene]
            order = np.lexsort((neighbors, -magnitude[neighbors]))[:top_k]
            for neighbor in neighbors[order]:
                selected.add(tuple(sorted((gene, int(neighbor)))))
    if not selected:
        raise ValueError(
            "No candidate edges passed; lower --min-correlation or inspect the data."
        )
    pairs = np.asarray(sorted(selected), dtype=np.int64)
    correlations = np.asarray(
        [np.dot(values[:, u], values[:, v]) for u, v in pairs], dtype=np.float64
    ).clip(-1, 1)
    directed = np.stack((pairs, pairs[:, ::-1]), axis=1).reshape(-1, 2).T
    return pairs, directed, correlations, int((norms == 0).sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--prepared-dir", type=Path, help="directory containing prepared.npz"
    )
    source.add_argument(
        "--training-csv",
        type=Path,
        help="training samples x genes; first column is sample ID",
    )
    parser.add_argument("--method", choices=("pearson", "spearman"), default="spearman")
    parser.add_argument("--min-correlation", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True, help="output network.npz")
    args = parser.parse_args()

    if args.prepared_dir is not None:
        input_path = args.prepared_dir / "prepared.npz"
        with np.load(input_path, allow_pickle=False) as data:
            indices = data["train_indices"]
            expression = data["expression"][indices]
            genes = data["genes"]
        training_only = "prepared_train_indices"
    else:
        input_path = args.training_csv
        expression, _, genes, _ = load_bio_data(input_path)
        indices = np.arange(expression.shape[0], dtype=np.int64)
        training_only = "user_supplied_training_csv"
    pairs, edge_index, correlations, constant_genes = correlation_edges(
        expression,
        method=args.method,
        min_correlation=args.min_correlation,
        top_k=args.top_k,
    )
    data_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        pairs=pairs,
        edge_index=edge_index,
        correlation=correlations,
        genes=np.asarray(genes, dtype=str),
        data_sha256=np.asarray(data_hash),
        train_indices=indices,
    )
    degrees = np.bincount(pairs.reshape(-1), minlength=len(genes))
    report = {
        "source": str(input_path.resolve()),
        "data_sha256": data_hash,
        "training_only": training_only,
        "training_samples": int(expression.shape[0]),
        "genes": len(genes),
        "undirected_edges": len(pairs),
        "method": args.method,
        "min_abs_correlation": args.min_correlation,
        "top_k_per_gene": args.top_k,
        "top_k_rule": "undirected union; final degree can exceed top_k",
        "constant_genes": constant_genes,
        "isolated_genes": int((degrees == 0).sum()),
        "interpretation": "candidate coexpression edges, not causal labels or directions",
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    with args.output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("source_gene", "target_gene", "correlation"))
        for (u, v), correlation in zip(pairs, correlations):
            writer.writerow((genes[u], genes[v], float(correlation)))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
