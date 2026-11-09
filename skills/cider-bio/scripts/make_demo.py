"""Create a small labelled, donor-grouped count dataset for testing the skill."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--h5ad", action="store_true", help="also write demo.h5ad")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    counts, records = [], []
    for donor in range(30):
        label = donor % 2
        donor_effect = rng.normal(0, 0.2)
        for cell in range(4):
            module = rng.normal(size=3)
            rates = np.exp(
                1.5
                + donor_effect
                + np.repeat(module * 0.3, 8)
                + np.r_[np.repeat(label * 0.8, 8), np.zeros(16)]
            )
            counts.append(rng.poisson(rates))
            records.append(
                {
                    "sample_id": f"donor_{donor:02d}_cell_{cell}",
                    "donor": f"donor_{donor:02d}",
                    "disease": "case" if label else "control",
                    "cell_type": "type_a" if cell < 2 else "type_b",
                }
            )
    obs = pd.DataFrame(records).set_index("sample_id")
    matrix = np.asarray(counts, dtype=np.float32)
    genes = [f"GENE_{index:02d}" for index in range(matrix.shape[1])]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix, index=obs.index, columns=genes).to_csv(
        args.output_dir / "expression.csv"
    )
    obs.to_csv(args.output_dir / "metadata.csv")
    if args.h5ad:
        import anndata as ad
        from scipy.sparse import csr_matrix

        ad.AnnData(
            X=csr_matrix(matrix), obs=obs, var=pd.DataFrame(index=genes)
        ).write_h5ad(args.output_dir / "demo.h5ad")
    print(f"Wrote 120 cells, 30 donors and 24 genes to {args.output_dir}")
    print("Synthetic API demonstration only; no biological or causal ground truth.")


if __name__ == "__main__":
    main()
