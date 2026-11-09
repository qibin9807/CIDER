# Validation-led parameter selection

## Scope and baseline

The bundled model is the formal teacher-free recursive CIDER. There is no
`teacher_stage`, offline deletion search, teacher forcing, or edge-ranking
pseudo-label dataset. Its frozen task classifier supplies differentiable
prediction feedback and online same-budget alternatives. Both roles are needed;
removing the classifier losses changes what the decomposition means.

Use gene correlation only for candidate topology. Do not train CIDER to copy
correlation ranks as if they were causal labels. A classifier must actually use
the learned nonnegative `edge_weight` in message passing for those losses to
reach the mask parameters.

## Search in two stages

1. Choose a small candidate gene panel/network and use `run_trial.py --task-only`
   to evaluate classifiers on validation macro-F1. Inspect majority baseline,
   balanced accuracy, per-class
   support and prediction collapse. For donor phenotypes, the effective sample
   count is donors, not cells. If the task has no usable validation signal,
   report the limitation rather than optimizing fidelity to a constant predictor.
2. Freeze a selected prepared dataset, network, and task run. Reuse its critic
   for CIDER comparisons via `--task-run`; otherwise two edge policies can appear
   different simply because they explain different prediction models.

An initial short search can vary one or two factors per stage:

| Component | Useful initial candidates |
| --- | --- |
| Selected genes | 64, 128; larger if independent sample count and memory justify it |
| Network | Spearman/Pearson; absolute cutoff 0.2–0.5; per-gene top-k 4–12 |
| Task GCN | hidden 32/64; 2/3 layers; learning rate 1e-3/3e-3 |
| CIDER | hidden 32/64; latent 8/16; decoder 32/64; 2 layers |
| CIDER optimization | learning rate 5e-4/1e-3; 10-epoch pilot, then 30 if useful |

These are starting points, not a mandatory grid. Estimate runtime from one
pilot, write the search budget and primary metrics, then choose subsequent
trials from validation evidence. Record unsuccessful trials too. Do not expand
compute indefinitely for a small validation gain; report what the budget tested.

Keep ranking temperature 1, soft-mask temperature 0.2, sparse-stage focus 2, and
one online alternative as the initial formal settings. Loss weights start at
fidelity 1, distillation 0.25, margin 0.25, step consistency 0.5, online
counterfactual 1, KL 1e-5, independence 0.01, invariance 0.01, entropy 0.001.
Changing a weight is a documented ablation, not a new external teacher.

The runner fixes task-checkpoint selection to validation macro-F1 and selects
CIDER by `0.7 * mean fidelity + 0.3 * 10%-30% fidelity`. Always report retained
graph label accuracy alongside fidelity. A method can preserve wrong predictions
perfectly. Use matched seeds and repeated independent seeds for promising
settings; describe uncertainty rather than selecting a seed using test.

## Configs and reuse

Pass a JSON config to `scripts/run_trial.py` and keep a separate output directory
per trial. The runner records its resolved config, validation history, selected
state, and input/network hashes. `--task-run work/run_a` reuses its task weights;
keep task-constructor settings compatible and the input/network unchanged.
The reused directory may come from `--task-only`, which does not fit CIDER or
produce `selected.pt`. Once the task validation result is adequate, omit
`--task-only` and pass `--task-run` to begin the policy search.
For concrete JSON and commands see [quickstart.md](quickstart.md).

## Test and interpretation

Write down the winning configuration, run ID, validation scores, selected seeds
and rationale before invoking `evaluate_run.py --split test`. The evaluation
script can technically evaluate any run: the agent is responsible for not
querying test repeatedly during selection. A test result may be reported as
negative evidence; it is not permission to tune until it looks positive.

Gene-named retained edges are relative to a particular fitted classifier and
the chosen candidate graph. Check stability across independent samples/seeds,
and distinguish correlation sign from causal effect sign. Do not present a
retained undirected coexpression edge as a regulatory direction or a measured
intervention effect. Real biological causal claims require identification
assumptions and independent intervention evidence.
