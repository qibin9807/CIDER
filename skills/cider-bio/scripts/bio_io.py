"""Read sample-by-gene expression and align sample metadata."""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


def _csv_frame(path, metadata=False):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        columns = next(csv.reader(handle))
    names = pd.Index(columns[1:])
    if names.has_duplicates or (names.str.strip() == "").any():
        raise ValueError(f"CSV column names must be nonempty and unique: {path}")
    # Read IDs as a regular string column; pandas can infer an unnamed index
    # back to integers even when read_csv(index_col=0, dtype=...) is supplied.
    frame = pd.read_csv(path, dtype=str if metadata else {0: str})
    identifiers = frame.pop(frame.columns[0])
    frame.columns = columns[1:]
    frame.index = pd.Index(identifiers, name=columns[0] or None)
    if frame.index.isna().any():
        raise ValueError(f"Sample IDs must not be missing: {path}")
    frame.index = frame.index.astype(str)
    if frame.index.has_duplicates:
        raise ValueError(f"Sample IDs must be unique: {path}")
    return frame


def load_bio_data(path, metadata=None, layer=None):
    """Return expression, obs, gene IDs, and input-alignment provenance."""
    path = Path(path)
    annotations = {
        "available_layers": [],
        "has_raw": False,
        "has_log1p_annotation": False,
    }
    if path.suffix.lower() == ".h5ad":
        import anndata as ad

        data = ad.read_h5ad(path)
        annotations = {
            "available_layers": list(data.layers),
            "has_raw": data.raw is not None,
            "has_log1p_annotation": "log1p" in data.uns,
        }
        expression = data.layers[layer] if layer is not None else data.X
        obs = data.obs.copy()
        genes = data.var_names.astype(str).to_numpy()
        if expression is None:
            raise ValueError("The selected h5ad expression matrix is empty.")
    elif path.suffix.lower() == ".csv":
        if layer is not None:
            raise ValueError("--layer applies to h5ad input, not CSV.")
        frame = _csv_frame(path)
        expression = frame.to_numpy(dtype=np.float64)
        obs = pd.DataFrame(index=frame.index)
        genes = frame.columns.astype(str).to_numpy()
    else:
        raise ValueError("Expression input must be a .h5ad or .csv file.")

    if obs.index.isna().any():
        raise ValueError("Expression sample IDs must not be missing.")
    obs.index = obs.index.astype(str)
    if obs.index.has_duplicates or (obs.index.str.strip() == "").any():
        raise ValueError("Expression sample IDs must be nonempty and unique.")
    if pd.Index(genes).has_duplicates or (pd.Index(genes).str.strip() == "").any():
        raise ValueError("Gene IDs must be nonempty and unique before preparation.")
    if obs.columns.has_duplicates:
        raise ValueError("Observation metadata columns must be unique.")
    if sparse.issparse(expression):
        expression = expression.tocsr()
    else:
        expression = np.asarray(expression)

    alignment = {
        "metadata_reordered": False,
        "metadata_rows_reordered": 0,
        **annotations,
    }
    if metadata is not None:
        extra = _csv_frame(metadata, metadata=True)
        missing = obs.index.difference(extra.index)
        unexpected = extra.index.difference(obs.index)
        if len(missing) or len(unexpected):
            raise ValueError(
                "Metadata sample IDs must match expression sample IDs exactly. "
                f"Missing metadata: {missing[:5].tolist()}; "
                f"metadata-only IDs: {unexpected[:5].tolist()}."
            )
        alignment["metadata_rows_reordered"] = int(
            np.count_nonzero(extra.index.to_numpy() != obs.index.to_numpy())
        )
        alignment["metadata_reordered"] = bool(alignment["metadata_rows_reordered"])
        extra = extra.loc[obs.index]
        for column in extra.columns:
            if column in obs:
                left = obs[column].astype("string")
                right = extra[column].astype("string")
                conflict = left.notna() & right.notna() & left.ne(right)
                if conflict.any():
                    raise ValueError(f"Conflicting h5ad and metadata values: {column}")
                obs[column] = left.fillna(right)
            else:
                obs[column] = extra[column]
    return expression, obs, np.asarray(genes, dtype=str), alignment
