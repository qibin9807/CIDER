# CIDER

Research code for recursive causal/non-causal graph decomposition.
Each round re-encodes the current residual graph and removes a small non-causal
edge subset, producing nested explanations from 90% down to 10% retained edges.
Training uses a frozen prediction model to assess whether each residual graph
preserves the target prediction. Given a target class, recursive edge selection
makes no prediction-model calls.

Choose either the agent workflow for biological expression data or the Python
workflow for direct use. Both use the same CIDER core. Source files are ordinary
Python modules; no package build or notebook is required.

```bash
git clone https://github.com/qibin9807/CIDER.git
cd CIDER
```

## For agents: install the biological-data skill

Ask your coding agent:

> Install the cider-bio skill from
> https://github.com/qibin9807/CIDER/tree/main/skills/cider-bio.

For Codex, you can also copy the entire skill folder manually:

```bash
mkdir -p ~/.agents/skills
cp -R skills/cider-bio ~/.agents/skills/
```

Codex supports user skills in `~/.agents/skills/` and repository skills in
`.agents/skills/`; see the official [skill installation and usage guide](https://learn.chatgpt.com/docs/build-skills).
Invoke `$cider-bio` explicitly or select it with `/skills`. If the new skill is
not listed, restart Codex. For another agent, copy the same complete folder to
that agent's documented skills directory.

The skill includes its own model code, helper scripts, and dependency list.
The agent can inspect h5ad/CSV data, prepare independent splits, select genes,
construct a training-only correlation network, and compare model parameters
using validation results. For example:

> Use $cider-bio on `study.h5ad`. Predict `obs.disease`, use counts from
> `layers.counts`, keep `obs.donor` disjoint across splits, and consider
> donor/cell-type pseudobulk. Choose genes, a correlation network, and CIDER
> parameters within six validation trials. Lock the selected run before the
> final test, then export gene-named explanations and rerun commands.

The agent should establish the actual outcome, count/transformation provenance,
and independent biological unit from your data. Gene selection, scaling, and
network construction use training samples only; test data are reserved for the
final selected analysis. Correlation networks define candidate topology, not
known biological causal edges or intervention effects.

See the [skill instructions](skills/cider-bio/SKILL.md) and
[runnable biological-data quickstart](skills/cider-bio/references/quickstart.md).

## For researchers: run the Python example

```bash
cd cider_minimal
python -m pip install -r requirements.txt
python example.py
```

The example generates independent toy splits, trains and validation-selects a
task classifier, trains CIDER, reloads the selected checkpoint, evaluates test
data, and saves all nine residual graphs to `example_output/`. Its generated
data demonstrate the API.

For your own PyTorch Geometric graphs, prepare a directory containing
`train.pt`, `validation.pt`, and `test.pt`, each holding a list of `Data` graphs:

```bash
python example.py --data-dir ../my_graphs --task-lr 0.001 \
  --epochs 30 --output-dir ../runs/my_study
```

Each graph needs node features `x`, a bidirectional undirected `edge_index`,
and a graph label `y`. The [Python guide](cider_minimal/README.md) describes
the tensor shapes, custom classifiers, pretrained-task reuse, and saved-model
reload. A custom classifier must use `edge_weight` in message passing so CIDER
can learn from masked prediction losses.

Direct inference with the selected Mutagenicity weights, from `cider_minimal/`:

```python
from experiments import load_best_model, load_graphs
model = load_best_model("mutag", device="cpu")
graph = load_graphs("../data/MUTAG/processed/testing.pt")[0]
explanations = model.explain_graph(graph)
print(explanations["0.1"].edge_index)
```

## Reproduce the experiments

Five selected weight files and the original processed splits are included.
From `cider_minimal/`, display the saved metrics or recompute them:

```bash
python show_results.py
python show_results.py -e nci1 mutag --split both
python show_results.py --recompute --split test
```

With no experiment argument, all three are shown. Recomputation defaults to
the included `../data`, CPU, batch size 128, and eight PyTorch threads.

| Experiment | Method | Test metric | Result |
| --- | --- | --- | ---: |
| NCI1 | Recursive CIDER | Mean prediction fidelity, 10%–90% retention | 87.40% |
| Mutagenicity (`mutag`) | Recursive CIDER | Mean prediction fidelity, 10%–90% retention | 92.47% |
| BA-2Motif | Supervised topology reference | Full-graph label accuracy | 100% |

BA-2Motif uses known motif-edge labels and is a separate supervised reference.
Its accuracy is not evidence that generic CIDER discovers biological causality.
Fidelity measures agreement with the frozen classifier; label accuracy is
reported separately. All included checkpoints were selected on validation.

The [Python guide](cider_minimal/README.md) records the model, losses, seeds,
selected epochs, environment, and evaluation protocol. The
[data notes](data/README.md) describe the frozen splits and NCI1 overlap removal.
The code is released under the [MIT License](LICENSE).

## Citation

If you use CIDER in your research, please cite:

Qibin Zhang, Chengshang Lyu, Lingxi Chen, Pei Chen, Rui Liu, and Luonan Chen
(2026). **CIDER: Counterfactual-invariant diffusion-based GNN explainer for causal
subgraph inference.** *The Innovation*, article 101607.
[https://doi.org/10.1016/j.xinn.2026.101607](https://doi.org/10.1016/j.xinn.2026.101607)

```bibtex
@article{zhang2026cider,
  title   = {{CIDER}: Counterfactual-invariant diffusion-based {GNN} explainer for causal subgraph inference},
  author  = {Zhang, Qibin and Lyu, Chengshang and Chen, Lingxi and
             Chen, Pei and Liu, Rui and Chen, Luonan},
  journal = {The Innovation},
  year    = {2026},
  pages   = {101607},
  doi     = {10.1016/j.xinn.2026.101607},
  url     = {https://doi.org/10.1016/j.xinn.2026.101607}
}
```
