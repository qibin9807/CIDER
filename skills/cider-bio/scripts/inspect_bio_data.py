"""Inspect biological expression and sample metadata before choosing a design."""

import argparse
import json
from pathlib import Path
import re

import numpy as np
from scipy import sparse

from bio_io import load_bio_data


def inspect_data(expression, obs, genes, alignment):
    values = expression.data if sparse.issparse(expression) else expression.reshape(-1)
    finite = np.isfinite(values)
    observed = values[finite]
    total = int(expression.shape[0] * expression.shape[1])
    nonzero = int(np.count_nonzero(values))
    integer_fraction = (
        float(np.isclose(observed, np.round(observed), atol=1e-6, rtol=0).mean())
        if observed.size
        else None
    )
    if not finite.all():
        hint = "contains_nonfinite_values"
    elif nonzero == 0:
        hint = "all_zero"
    elif observed.min() < 0:
        hint = "continuous_with_negative_values"
    elif integer_fraction >= 0.999:
        hint = "nonnegative_integer_like_counts"
    else:
        hint = "nonnegative_continuous_or_log1p"
    columns = {}
    candidates = []
    for column in obs.columns:
        series = obs[column]
        counts = series.astype("string").fillna("<missing>").value_counts()
        columns[str(column)] = {
            "dtype": str(series.dtype),
            "missing": int(series.isna().sum()),
            "n_unique": int(series.nunique()),
            "value_counts": {
                str(key): int(value) for key, value in counts.head(20).items()
            },
            "value_counts_truncated": len(counts) > 20,
        }
        if re.search(
            r"donor|patient|subject|participant|individual|specimen|sample|group",
            str(column),
            re.I,
        ):
            candidates.append(str(column))
    return {
        "shape": [int(value) for value in expression.shape],
        "sparse_storage": bool(sparse.issparse(expression)),
        "zero_fraction": 1 - nonzero / total if total else None,
        "nonfinite_values": int((~finite).sum()),
        "integer_like_fraction_of_stored_values": integer_fraction,
        "scale_hint": hint,
        "scale_note": "Hints do not determine provenance; choose --input-scale explicitly.",
        "available_layers": alignment["available_layers"],
        "has_raw": alignment["has_raw"],
        "has_log1p_annotation": alignment["has_log1p_annotation"],
        "sample_ids_preview": obs.index[:10].tolist(),
        "genes_preview": genes[:10].tolist(),
        "obs_columns": columns,
        "group_column_candidates": candidates,
        "alignment": alignment,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--layer")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    expression, obs, genes, alignment = load_bio_data(
        args.input, args.metadata, args.layer
    )
    summary = inspect_data(expression, obs, genes, alignment)
    summary["source"] = {
        "input": str(args.input.resolve()),
        "metadata": str(args.metadata.resolve()) if args.metadata else None,
        "layer": args.layer,
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
