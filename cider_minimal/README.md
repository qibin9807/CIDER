# CIDER research code

This source bundle contains the formal teacher-free recursive CIDER model,
selected NCI1 and Mutagenicity weights, and a separate supervised BA-2Motif
reference. Run the Python files directly from this directory.

## Environment

```bash
pip install -r requirements.txt
```

Checked with Python 3.12.13, PyTorch 2.7.1+cu128, and PyG 2.6.1. Archived metrics
were evaluated on CPU with eight PyTorch threads. Reuse the same versions and
device when comparing numerical results.

The compact model was checked against the main implementation: all cleaned
validation graph outputs, recursive edge selections, training losses, gradients,
and an optimizer epoch matched exactly in the same environment. Recomputing
both reported splits with the bundled weights also matched the archived metrics.

## Show or recompute results

```bash
python show_results.py                         # all three, test results
python show_results.py -e nci1                  # selected experiment
python show_results.py -e nci1 mutag --split both
python show_results.py --recompute --data-root ../data --split test --torch-threads 8
```

The default reads `results.json` and needs only Python's standard library.
`--recompute` loads the bundled weights and evaluates the original processed
data at `../data/{NCI1,MUTAG,BA-2motif}/processed/`. Supply `--data-root` if that
directory is elsewhere. The original processed splits must be retained; newly
randomized splits will produce different results.
Recomputation defaults to CPU, batch size 128, and eight PyTorch threads;
`--device`, `--batch-size`, and `--torch-threads` can change these settings.

| Experiment | Model | Selected seed / epoch | Validation / test graphs | Test mean fidelity |
| --- | --- | --- | --- | ---: |
| `nci1` | Teacher-free recursive CIDER | 1 / 20 | 486 / 485 | 87.40% |
| `mutag` | Teacher-free recursive CIDER | 0 / 10 | 500 / 500 | 92.47% |
| `ba2` | Supervised topology reference | 0 / 60 | 100 / 100 | Separate reference |

NCI1 evaluation removes cross-split attributed-graph isomorphisms, without
using labels, to match the archived results. `mutag` denotes the 4,337-graph
Mutagenicity dataset. BA-2Motif uses known motif-edge labels during training;
its 100% test accuracy is a dataset-specific supervised reference.

`params/{nci1,mutag}/best.pt` contains the selected CIDER policy and training
metadata. The matching frozen classifier is in `task_model.pt` alongside it.
`params/ba2/best.pt` contains the reference graph and edge heads. All selected
checkpoints were chosen on validation data.

```python
from experiments import load_best_model

model = load_best_model("nci1", device="cpu")
# Other choices: "mutag", "ba2".
```

## Model and training settings

Every round re-encodes the residual graph, estimates competing causal and
non-causal edge distributions, and removes the highest non-causal edge pairs.
The retained graphs are nested from 90% down to 10%. For `E` undirected pairs,
retention `r` keeps `max(1, ceil(r * E))` pairs; reverse directions stay together.

No offline teacher stages, teacher forcing, or edge pseudo-label supervision
are used by the recursive model. The frozen task classifier supplies training
feedback through fidelity, distillation, margin, step consistency, and online
same-budget counterfactual comparisons. With an explicit target class,
`explain_graph` makes no classifier calls during recursive deletion. Without a
target, one full-graph classifier call chooses the class before deletion.

The selected models use hidden widths 64 for NCI1 and 32 for Mutagenicity,
latent width 16, two encoder layers, decoder width 64, zero task-node context,
and no dropout. Training uses 30 epochs, batch size 128, Adam learning rate
1e-3, validation every five epochs, ranking temperature 1, soft-mask
temperature 0.2, sparse-stage focus 2, and one online alternative per round.
Loss weights are:

```text
fidelity 1         distillation 0.25     margin 0.25
step consistency 0.5                    online counterfactual 1
KL 1e-5            independence 0.01     invariance 0.01     entropy 0.001
```

Validation selects the maximum of `0.7 * mean_fidelity + 0.3 * fidelity_10%-30%`.
Fidelity measures agreement with the frozen classifier; label accuracy is
reported separately. This defines prediction-relative explanations, not
ground-truth scientific causal effects.

## Run the end-to-end example

```bash
python example.py
```

The default creates independent synthetic splits (32 training, eight validation,
eight test graphs) and runs task-classifier training, validation selection,
teacher-free CIDER training, checkpoint reload, test evaluation, and single-graph
explanation. It uses CPU, seed 0, 20 task epochs at learning rate 0.01, five CIDER
epochs, batch size 16, validation every five CIDER epochs, and one PyTorch thread.
The generated data demonstrate the API; use
`show_results.py --recompute` for the paper's benchmark results.

The example writes the following files to `example_output/`:

| File | Contents |
| --- | --- |
| `model.pt` | Selected full model state, constructor settings, epoch, and validation metrics |
| `task_model.pt` | Plain task-classifier `state_dict`, reusable with `--task-checkpoint` |
| `metrics.json` | Test fidelity and label accuracy at all nine retention ratios |
| `explanations.pt` | Nine residual PyG graphs, keyed by `"0.9"`, ..., `"0.1"` |
| `example_graph.pt` | Original test graph used for the explanation |

## Use your own graph data

Prepare independent training, validation, and test lists of PyG `Data` graphs
and save them as `train.pt`, `validation.pt`, and `test.pt` in one directory:

```bash
python example.py --data-dir my_graphs --task-lr 0.001 --epochs 30 --output-dir my_run
python example.py --data-dir my_graphs --task-checkpoint my_run/task_model.pt \
  --epochs 30 --output-dir my_pretrained_run
```

Tune the classifier's schedule for your data with `--task-epochs` and
`--task-lr`. `--epochs` and `--validation-interval` control CIDER training.

Each graph needs floating-point `x` of shape `[num_nodes, num_features]`, long
`edge_index` of shape `[2, num_directed_edges]`, and one label `y` in
`0, ..., num_classes - 1`. Graphs must contain nodes and edges. Store both
directions of each undirected edge. For example, this is a three-node graph:

```python
import torch
from torch_geometric.data import Data

graph = Data(
    x=torch.tensor([[1., 0.], [0., 1.], [1., 1.]]),
    edge_index=torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]]),
    y=torch.tensor([1]),
)
```

After building one graph per sample and forming disjoint `train_graphs`,
`validation_graphs`, and `test_graphs` lists, save them with:

```python
from pathlib import Path

directory = Path("my_graphs")
directory.mkdir(exist_ok=True)
for filename, graphs in (
    ("train.pt", train_graphs),
    ("validation.pt", validation_graphs),
    ("test.pt", test_graphs),
):
    torch.save(graphs, directory / filename)
```

The reader also accepts original PyG collated `(data, slices)` files. Optional
`edge_attr` is passed to custom classifiers. The example's
`--task-checkpoint` expects a plain `GcnTaskModel.state_dict()` with the same
architecture as its classifier constructor. To use another trained classifier,
replace its construction in both the training/loading and checkpoint-reload
sections of `example.py`, and save its constructor arguments in `task_config`.

A custom classifier must accept `x`, `edge_index`, `batch`, and optional
`edge_attr`/`edge_weight`, returning `[num_graphs, num_classes]` logits. It must
use `edge_weight` in message passing so the masked training losses can train
CIDER's edge policy. CIDER freezes the selected classifier during its own
training. Set the random seed before constructing both models; keep
validation and test independent of training.

## Reload a model and explain a new graph

The example reloads its checkpoint before test evaluation. The same steps can
be used in your own code:

```python
import torch

from cider import GcnTaskModel, IterativeCIDER

saved = torch.load("example_output/model.pt", map_location="cpu", weights_only=False)
task_model = GcnTaskModel(**saved["task_config"])
model = IterativeCIDER(
    saved["task_config"]["input_dim"], task_model, **saved["cider_config"]
)
model.load_state_dict(saved["model_state_dict"])
model.eval()

graph = torch.load(
    "example_output/example_graph.pt", map_location="cpu", weights_only=False
)
explanations = model.explain_graph(graph)
print(explanations["0.1"].edge_index)
```

To choose the class explicitly, use `model.explain_graph(graph, target=1)`.
That call makes no classifier queries during recursive deletion. Preserve the
default retention ratios to run all nine trained stages. Retraining varies
with random seeds and device arithmetic, so
cross-device bitwise equality is not assumed.
