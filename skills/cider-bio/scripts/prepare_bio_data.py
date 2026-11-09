"""Prepare patient-disjoint expression splits and training-only gene selection."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.utils.sparsefuncs import mean_variance_axis

from bio_io import load_bio_data


def aggregate_samples(expression, obs, columns, label_column, group_column, scale):
    codes, keys = pd.factorize(pd.MultiIndex.from_frame(obs[columns]), sort=False)
    sizes = np.bincount(codes)
    checked_columns = list(
        dict.fromkeys([label_column] + ([group_column] if group_column else []))
    )
    unique_values = obs.groupby(codes, sort=False)[checked_columns].nunique()
    if (unique_values > 1).any().any():
        raise ValueError(
            "Each aggregation group must have one label and one patient/donor group."
        )
    indicator = sparse.csr_matrix(
        (np.ones(len(obs)), (codes, np.arange(len(obs)))),
        shape=(len(sizes), len(obs)),
    )
    expression = indicator @ expression
    if scale != "counts":
        expression = (
            sparse.diags(1 / sizes) @ expression
            if sparse.issparse(expression)
            else expression / sizes[:, None]
        )
    first = np.unique(codes, return_index=True)[1]
    obs = obs.iloc[first].copy()
    obs.index = pd.Index(
        [
            json.dumps(
                [str(value) for value in key], ensure_ascii=False, separators=(",", ":")
            )
            for key in keys
        ]
    )
    return expression, obs, sizes


def split_samples(labels, groups, validation_fraction, test_fraction, seed):
    indices = np.arange(len(labels))
    held_out_fraction = validation_fraction + test_fraction
    if groups is None:
        training, held_out = train_test_split(
            indices, test_size=held_out_fraction, stratify=labels, random_state=seed
        )
        validation, test = train_test_split(
            held_out,
            test_size=test_fraction / held_out_fraction,
            stratify=labels[held_out],
            random_state=seed,
        )
    else:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=held_out_fraction, random_state=seed
        )
        training, held_out = next(splitter.split(indices, labels, groups))
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=test_fraction / held_out_fraction, random_state=seed
        )
        validation_positions, test_positions = next(
            splitter.split(held_out, labels[held_out], groups[held_out])
        )
        validation, test = held_out[validation_positions], held_out[test_positions]
    if len(np.unique(labels[training])) != len(np.unique(labels)):
        raise ValueError(
            "The training split lacks a class. Review the patient/group design, "
            "class counts, and seed instead of fitting this split."
        )
    return {"train": training, "validation": validation, "test": test}


def normalize_expression(expression, scale):
    expression = expression.astype(np.float64)
    if scale == "counts":
        totals = np.asarray(expression.sum(axis=1)).reshape(-1)
        if (totals <= 0).any():
            raise ValueError(
                "Count samples with zero library totals must be reviewed before training."
            )
        if sparse.issparse(expression):
            expression = expression.multiply((1e4 / totals)[:, None]).tocsr()
            expression.data = np.log1p(expression.data)
        else:
            expression = np.log1p(expression * (1e4 / totals)[:, None])
    return expression.astype(np.float32)


def prepare(args):
    if not (
        0 < args.validation_fraction
        and 0 < args.test_fraction
        and args.validation_fraction + args.test_fraction < 1
    ):
        raise ValueError(
            "Validation and test fractions must be positive and sum to less than one."
        )
    if args.max_genes < 1:
        raise ValueError("--max-genes must be positive.")
    expression, obs, genes, alignment = load_bio_data(
        args.input, args.metadata, args.layer
    )
    values = expression.data if sparse.issparse(expression) else expression.reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("Expression values must be finite before preparation.")
    if args.input_scale == "counts" and (values < 0).any():
        raise ValueError("--input-scale counts requires nonnegative expression.")
    original_shape = list(expression.shape)
    aggregate_by = (
        [column.strip() for column in args.aggregate_by.split(",")]
        if args.aggregate_by
        else []
    )
    needed = list(
        dict.fromkeys(
            [args.label_column]
            + ([args.group_column] if args.group_column else [])
            + aggregate_by
        )
    )
    for column in needed:
        if column not in obs:
            raise ValueError(f"Metadata column not found: {column}")
        if obs[column].isna().any() or obs[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Metadata column has missing values: {column}")
    aggregation_sizes = None
    if aggregate_by:
        expression, obs, aggregation_sizes = aggregate_samples(
            expression,
            obs,
            aggregate_by,
            args.label_column,
            args.group_column,
            args.input_scale,
        )
    classes, labels = np.unique(
        obs[args.label_column].astype(str).to_numpy(), return_inverse=True
    )
    if len(classes) < 2:
        raise ValueError(
            "A graph-classification task requires at least two label classes."
        )
    patient_groups = (
        obs[args.group_column].astype(str).to_numpy() if args.group_column else None
    )
    splits = split_samples(
        labels, patient_groups, args.validation_fraction, args.test_fraction, args.seed
    )
    expression = normalize_expression(expression, args.input_scale)

    # Work on sparse training moments; densify only the selected gene columns.
    training = expression[splits["train"]].astype(np.float64)
    if sparse.issparse(training):
        _, variance = mean_variance_axis(training, axis=0)
        nonzero = np.asarray((training != 0).sum(axis=0)).reshape(-1)
    else:
        variance = training.var(axis=0)
        nonzero = np.count_nonzero(training, axis=0)
    eligible = np.flatnonzero((nonzero > 0) & (variance > 0))
    if not len(eligible):
        raise ValueError(
            "No nonzero, positive-variance genes occur in the training split."
        )
    selected = eligible[
        np.argsort(-variance[eligible], kind="stable")[: args.max_genes]
    ]
    expression = expression[:, selected]
    if sparse.issparse(expression):
        expression = expression.toarray()
    expression = np.asarray(expression, dtype=np.float32)
    training = expression[splits["train"]].astype(np.float64)
    scaler_mean = training.mean(axis=0).astype(np.float32)
    scaler_std = training.std(axis=0).astype(np.float32)
    sample_ids = obs.index.to_numpy(dtype=str)
    groups = patient_groups if patient_groups is not None else sample_ids

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "prepared.npz",
        expression=expression,
        labels=labels.astype(np.int64),
        train_indices=splits["train"].astype(np.int64),
        validation_indices=splits["validation"].astype(np.int64),
        test_indices=splits["test"].astype(np.int64),
        genes=genes[selected].astype(str),
        sample_ids=sample_ids,
        groups=np.asarray(groups, dtype=str),
        mean=scaler_mean,
        std=scaler_std,
        classes=classes.astype(str),
    )
    summary = {
        "source": {
            "input": str(args.input.resolve()),
            "metadata": str(args.metadata.resolve()) if args.metadata else None,
            "layer": args.layer,
            "input_scale": args.input_scale,
            "aggregate_by": aggregate_by,
        },
        "shape": {
            "input_samples": int(original_shape[0]),
            "input_genes": int(original_shape[1]),
            "samples": int(expression.shape[0]),
            "genes": int(expression.shape[1]),
        },
        "label_column": args.label_column,
        "group_column": args.group_column,
        "class_names": classes.tolist(),
        "split_method": "group_shuffle" if patient_groups is not None else "stratified",
        "split_counts": {name: int(len(indices)) for name, indices in splits.items()},
        "class_counts_by_split": {
            name: {
                label: int(np.count_nonzero(labels[indices] == index))
                for index, label in enumerate(classes)
            }
            for name, indices in splits.items()
        },
        "split_group_counts": {
            name: int(len(np.unique(groups[indices])))
            for name, indices in splits.items()
        },
        "selected_genes": genes[selected].tolist(),
        "gene_selection": {
            "fit": "training_only",
            "method": "top_variance",
            "eligible_genes": int(len(eligible)),
        },
        "normalization": "library_size_1e4_log1p_all_input_genes"
        if args.input_scale == "counts"
        else "unchanged",
        "standardization": {"fit": "training_only", "ddof": 0},
        "seed": args.seed,
        "sample_tracking": {
            "metadata_alignment": alignment,
            "excluded_samples": [],
            "sample_order": "first_observed_aggregation_group"
            if aggregate_by
            else "input_order",
            "aggregation_reduction": ("sum" if args.input_scale == "counts" else "mean")
            if aggregate_by
            else None,
            "aggregation_group_sizes": aggregation_sizes.tolist()
            if aggregation_sizes is not None
            else None,
        },
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    (args.output_dir / "prepare.json").write_text(text)
    print(text, end="")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--group-column")
    parser.add_argument("--layer")
    parser.add_argument(
        "--input-scale", required=True, choices=("counts", "log1p", "continuous")
    )
    parser.add_argument(
        "--aggregate-by", help="comma-separated obs columns, e.g. donor,cell_type"
    )
    parser.add_argument("--max-genes", type=int, default=128)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
