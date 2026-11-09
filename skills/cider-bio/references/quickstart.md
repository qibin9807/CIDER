# Run the skill helpers

The agent should resolve this directory from the loaded `SKILL.md`. The examples
below use a shell variable only to keep paths short:

```bash
CIDER_SKILL_DIR=/path/to/cider-bio
python -m pip install -r "$CIDER_SKILL_DIR/requirements.txt"
```

Use an existing compatible environment or a project-local virtual environment.
The scripts contain the required CIDER core, so a checkout of the original
repository is not required. Generated artifacts go into `work/`, not the skill.

## A complete small h5ad demonstration

```bash
python "$CIDER_SKILL_DIR/scripts/make_demo.py" --output-dir work/demo --h5ad
python "$CIDER_SKILL_DIR/scripts/inspect_bio_data.py" \
  --input work/demo/demo.h5ad --output work/inspection.json
python "$CIDER_SKILL_DIR/scripts/prepare_bio_data.py" \
  --input work/demo/demo.h5ad --label-column disease --group-column donor \
  --aggregate-by donor,cell_type --input-scale counts \
  --max-genes 24 --seed 0 --output-dir work/prepared
python "$CIDER_SKILL_DIR/scripts/build_network.py" \
  --prepared-dir work/prepared --method spearman \
  --min-correlation 0.3 --top-k 8 --output work/network.npz
python "$CIDER_SKILL_DIR/scripts/run_trial.py" \
  --prepared-dir work/prepared --network work/network.npz \
  --config "$CIDER_SKILL_DIR/assets/pilot.json" --output-dir work/pilot
```

`make_demo.py` also writes `expression.csv` and `metadata.csv`. To test the CSV
path, replace the preparation input with
`--input work/demo/expression.csv --metadata work/demo/metadata.csv` and keep the
same grouping and scale arguments. These artificial data are for execution
checks, not benchmark accuracy or validation of biological causality.

`assets/pilot.json` is an editable starting point. The runner merges unspecified
settings with its defaults and records the complete resolved config. Inspect
the task validation performance and the majority baseline before interpreting
retained edges. Expand or stop a real search according to validation evidence,
the independent sample count, and the runtime observed in this pilot.

## Let the agent propose another trial

For real model selection, train the task classifier first without fitting CIDER:

```bash
python "$CIDER_SKILL_DIR/scripts/run_trial.py" \
  --prepared-dir work/prepared --network work/network.npz \
  --config "$CIDER_SKILL_DIR/assets/pilot.json" --task-only --output-dir work/task_pilot
```

Compare its validation class metrics and baseline with other task-only trials.
If the classifier is usable, pass the winning directory through `--task-run` to
fit CIDER. If it collapses, continue the bounded task-model search or report the
limitation; high explanation fidelity alone is not a reason to proceed.

For example, the agent can write a second config containing:

```json
{
  "seed": 0,
  "cider": {
    "hidden_channels": 64,
    "latent_channels": 16,
    "learning_rate": 0.0005,
    "epochs": 10,
    "validation_interval": 5
  }
}
```

Then reuse the selected task classifier for a matched comparison:

```bash
python "$CIDER_SKILL_DIR/scripts/run_trial.py" \
  --prepared-dir work/prepared --network work/network.npz \
  --config work/second.json --task-run work/task_pilot --output-dir work/second
```

Avoid including different task settings in this CIDER-only config. Changing
the prepared genes or candidate graph requires a new classifier comparison,
not `--task-run` reuse. See [tuning.md](tuning.md) for selection criteria and
finite search budgets. Multiple seeds and alternative networks are chosen by
the agent; the user need not prescribe the hyperparameters.

## Lock the selected run, then evaluate

After the agent records its selected run and rationale, evaluate that run:

```bash
python "$CIDER_SKILL_DIR/scripts/evaluate_run.py" \
  --run-dir work/pilot --prepared-dir work/prepared --network work/network.npz \
  --split test --output-dir work/final_test
```

Replace `work/pilot` with the actual validation-selected run. For pre-selection
diagnostics, `--split validation` is available; do not invoke test while choosing
between candidate configs.

Outputs include preparation/network provenance, resolved trial configs,
validation histories, `task.pt`, `selected.pt`, and requested-split metrics.
`selected_edges.csv` names the sample, class target, retention ratio, gene
endpoints, and original correlation for a small number of explained samples.
`--max-explanations` changes the export count; prediction metrics still use the
whole requested split. Explain every trained retention stage, not a shortcut
straight from the full graph to 10%.

## Example request to the agent

> Use $cider-bio on `/data/study.h5ad`. Predict `obs.disease`; keep `obs.donor`
> disjoint, use the raw counts in `layers.counts`, and consider donor/cell-type
> pseudobulk. Choose the gene panel, correlation network and CIDER parameters
> from validation results within six pilot trials. Keep the final test out of
> parameter selection, then export gene-named explanations and rerun commands.

If the dataset has different biological semantics, ask about the target and
replication unit rather than silently applying this illustrative design.
