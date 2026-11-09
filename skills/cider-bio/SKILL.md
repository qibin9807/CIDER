---
name: cider-bio
description: Prepare h5ad or gene-expression tables, construct training-only gene correlation networks, and tune teacher-free CIDER to explain phenotype or cell-class predictions. Use for agent-led CIDER biological-data workflows, parameter selection, and gene-edge interpretation; not for generic differential expression or estimating biological intervention effects from correlations alone.
---

# CIDER for biological expression data

Turn biological expression data into a reproducible, validation-selected CIDER
analysis. Choose preprocessing and hyperparameters on the user's behalf. Use the
bundled teacher-free model in `assets/cider/cider.py`; it is the same core as the
formal research bundle. Do not substitute the archived teacher-supervised model.

## Establish the biological task

Inspect inputs with `scripts/inspect_bio_data.py` before deciding the analysis
unit. Ask only for missing scientific choices: the outcome column, the unit of
independent replication, and whether the matrix contains counts or transformed
expression. Do not ask the user to choose learning rates or network thresholds.

- Rows become cells or biological samples; genes become nodes. Each graph uses
  the same training-derived candidate network and sample-specific expression.
- For patient/donor phenotypes, use `--group-column` to keep each individual in
  one split. Consider pseudobulk with `--aggregate-by donor,cell_type` when it
  fits the outcome; aggregation groups must have one consistent outcome.
- For other designs, explicitly record what is independent. Random cell splits
  are not evidence of generalization to unseen patients.
- A phenotype covariate, treatment, or batch column is not automatically the
  correct outcome. Inspect `obs`; do not silently invent a target.

Read [references/data-workflow.md](references/data-workflow.md) for input formats,
transforms, sample grouping, and executable preparation/network commands.

## Prepare and build the candidate graph

Use `prepare_bio_data.py` with an explicit input scale. The script splits before
gene selection and fits gene standardization on training rows only. Preserve
`prepare.json`, gene IDs, class names, and split indices. It handles sparse h5ad
without making the entire original matrix dense.

Use `build_network.py --prepared-dir ...` for CIDER. It reads only the stored
training indices. Choose Pearson or Spearman and an absolute-correlation cutoff
and per-gene top-k; inspect isolated genes and edge density before training.
Candidates selected by either endpoint are merged into undirected pairs.

These are coexpression candidates, not causal edge labels. Signed correlations
are saved as metadata; the bundled GCN uses candidate topology and learned
nonnegative masks, not negative correlations as normalization weights. Node
features combine training-standardized expression and gene identity.

If preprocessing, selected genes, or network construction changes, create a new
prepared/network artifact and retrain its task classifier. Never fit genes,
correlations, or scaling on validation/test to improve results.

## Let the agent select parameters

Read [references/tuning.md](references/tuning.md), then use `run_trial.py` for
small, recorded validation trials. The runner does not evaluate test data.

1. Start with a pilot sized to the available device and independent sample
   count. Use its runtime and memory to choose a bounded initial search, such
   as six trials, rather than an unbounded Cartesian sweep.
2. Compare prediction models with `run_trial.py --task-only` using validation
   macro-F1 and inspect class-wise performance against the majority baseline.
   Network/feature alternatives belong to this stage. Fidelity cannot rescue a
   poor classifier; do not spend the CIDER search on a collapsed predictor.
3. Freeze the selected candidate graph and critic. Reuse it with `--task-run`
   while comparing CIDER settings. Keep the official teacher-free objective;
   adjust widths, learning rate, regularization or online-comparison settings
   through trial configs, not through external edge pseudo-labels.
4. Select CIDER on `0.7 * mean fidelity + 0.3 * 10%-30% fidelity`, reporting label
   accuracy separately. If a user chooses another validation objective, record
   it explicitly rather than silently changing the method.
5. Check promising settings across multiple seeds when independent samples and
   compute allow. Lock the chosen run IDs and configuration in the report before
   using `evaluate_run.py --split test`. Do not resume tuning from test results.

Do not overwrite different trials into one directory. Preserve configs,
validation metrics, selected states, seeds, input/network hashes, environment,
and gene mappings. New-data results are not the archived NCI1/Mutagenicity
benchmark; the synthetic demo is an interface check only.

## Explain and hand off

Use `evaluate_run.py` to export requested-split metrics and gene-named retained
edges. Keep all nine recursive stages: requesting only 10% skips trained stages
and changes the method. With an explicit target, recursive edge selection makes
zero critic calls; without one, a single full-graph call chooses the target.

Report the biological unit and group split; transformation and training-only
gene/network choices; the validation search and winning settings; task-model
macro-F1/accuracy and CIDER fidelity/accuracy; gene-edge findings and their
sample/seed stability; artifact paths and exact rerun commands.

Call outputs prediction-relative candidate explanations. Observational
correlation plus prediction fidelity does not identify biological causality,
edge direction, or intervention-effect magnitude. Those require an explicit
causal estimand, suitable data, and a separate effect-estimation study.

## Environment and quick check

Run in an existing compatible Python environment, or create a project-local
environment with `requirements.txt`. No repository checkout or private path is
needed: scripts resolve the bundled core relative to this skill directory.
Read [references/quickstart.md](references/quickstart.md) for a complete runnable
CSV/h5ad demo, trial config, evaluation, and an example invocation for real data.
