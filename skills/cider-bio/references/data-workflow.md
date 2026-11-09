# Expression data and candidate networks

Set `CIDER_SKILL_DIR` to the directory containing `SKILL.md`. Choose an output
workspace separate from the input files and the installed skill.

## Inspect and identify the task

```bash
python "$CIDER_SKILL_DIR/scripts/inspect_bio_data.py" --input study.h5ad
```

h5ad uses observations as rows and `var_names` as gene IDs. Inspect `obs`, layers,
sparsity, missing labels, and donor/batch columns. `--layer counts` selects a
named layer; the default is `X`. Counts in a layer and already-log-transformed
`X` must not be confused.

The reader exposes `X` and named layers. If the desired matrix is in
`adata.raw`, export `adata.raw.to_adata()` to a separate working h5ad and inspect
it before preparation, preserving the original file. `raw` is a storage name,
not proof that those values are untransformed counts.

CSV uses sample rows and gene columns, with sample IDs in the first column.
Supply a metadata CSV with the same sample IDs in its first column:

```bash
python "$CIDER_SKILL_DIR/scripts/inspect_bio_data.py" \
  --input expression.csv --metadata metadata.csv
```

The scripts align metadata by ID, not by row position. Duplicate sample or gene
identifiers and missing required metadata must be resolved, not silently merged.
Labels are classification outcomes; continuous outcomes require a separate
regression extension rather than arbitrary discretization.

## Prepare training, validation, and test

```bash
python "$CIDER_SKILL_DIR/scripts/prepare_bio_data.py" \
  --input study.h5ad --layer counts \
  --label-column disease --group-column donor \
  --input-scale counts --max-genes 128 --seed 0 --output-dir work/prepared
```

For patient phenotypes with multiple cell types, consider biological pseudobulk:

```bash
python "$CIDER_SKILL_DIR/scripts/prepare_bio_data.py" \
  --input study.h5ad --label-column disease --group-column donor \
  --aggregate-by donor,cell_type --input-scale counts \
  --max-genes 128 --seed 0 --output-dir work/pseudobulk
```

Counts are summed during aggregation, then library-size normalized to 10,000
and log1p transformed. `log1p` and `continuous` inputs are not normalized again;
their optional aggregation uses means. Aggregation is a modelling choice, not a
universal prerequisite. Case/control outcome and donor identity must be constant
within an aggregated observation.

Pooling different cell types can make coexpression primarily reflect cell-type
differences. For a cell-type-specific biological question, subset that stratum
into a separate working input before preparation, or run separate analyses.
Do not interpret a pooled network as a within-cell-type regulatory network.

Split by donor before selecting genes or learning normalization statistics.
Default validation/test fractions are 0.2 each. Group splitting preserves whole
groups, so actual fractions can differ. Inspect each split's group and class
counts; predeclare or revise the split design before model selection if the
independent cohort is too small. Stratification uses labels for split design,
not test predictions.

The supplied metric helpers weight graph rows equally. Cell-level or
donor/cell-type-level accuracy is not automatically donor-level accuracy. For a
donor-level endpoint, use one graph per donor or explicitly aggregate predictions
by donor and report that metric; do not relabel a per-cell score as patient-level.

`--max-genes` ranks training-set variance after the selected transform. This is
a simple high-variance filter, not Scanpy/Seurat's mean-corrected HVG method.
Training-only mean/std standardize the selected genes. Never recompute these
statistics using the held-out rows. Selecting a biologically predefined gene
panel can be added by adapting this stage while preserving split boundaries.

`prepared.npz` stores expression, integer labels/classes, genes/sample IDs,
groups, split indices, and training mean/std. `prepare.json` records how these
were obtained. Graphs are built lazily as
`x = [standardized expression, one-hot gene identity]`; shared expression and
topology are stored once, rather than serializing repeated dense graph lists.
One-hot width grows with gene count. Estimate batch memory before increasing
the initial 64–128-gene pilot to larger panels.

## Construct a training-only correlation graph

```bash
python "$CIDER_SKILL_DIR/scripts/build_network.py" \
  --prepared-dir work/prepared --method spearman \
  --min-correlation 0.3 --top-k 8 --output work/network.npz
```

Spearman is a useful initial robust coexpression candidate; Pearson can be
appropriate for approximately linear transformed expression. Select alternatives
using training diagnostics and validation performance. The builder uses only
`train_indices`, removes self-connections, keeps positive and negative
correlations by magnitude, and includes a pair if either gene selects the other.
Final degree can exceed top-k. It does not invent edges to connect isolates.

The output includes directed paired indices, canonical undirected pairs, signed
correlations, gene order, training indices, and the input hash. Its CSV sidecar
maps endpoints to gene IDs; its JSON sidecar reports density and isolated genes.
Changing a held-out expression row cannot change the correlations or selected
edges, although the whole-input provenance hash will change.

For standalone network construction from a training-only table:

```bash
python "$CIDER_SKILL_DIR/scripts/build_network.py" \
  --training-csv train_expression.csv --method pearson \
  --min-correlation 0.4 --top-k 8 --output work/coexpression.npz
```

Use the `--prepared-dir` path for CIDER trials; their input hashes and training
indices are checked. A standalone training CSV graph is an independent network
artifact and is not automatically interchangeable with a prepared dataset.
