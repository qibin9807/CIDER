# Frozen experiment data

These nine processed files preserve the graph order and train/validation/test
assignments used by the included selected checkpoints. They are the existing
project's frozen processed NCI1, Mutagenicity and BA-2Motif datasets. No new
biological user data are included.

| Directory | Dataset | Training | Stored validation | Stored test | Evaluated validation / test |
| --- | --- | ---: | ---: | ---: | ---: |
| `NCI1/processed` | NCI1 | 3,110 | 500 | 500 | 486 / 485 |
| `MUTAG/processed` | Mutagenicity | 3,337 | 500 | 500 | 500 / 500 |
| `BA-2motif/processed` | BA-2Motif | 800 | 100 | 100 | 100 / 100 |

Each directory contains `training.pt`, `evaluation.pt`, and `testing.pt` in PyG
collated `(data, slices)` format. `cider_minimal/experiments.py` reads them
without needing dataset download or processing classes.

NCI1 has attributed graph isomorphisms shared across stored splits. Evaluation
filters validation against training, then test against training and retained
validation, without using labels for matching. This removes 14 validation graphs
and 15 test graphs. The same check finds no cross-split overlap in Mutagenicity.
The cleaned sizes and fingerprints also remain in the checkpoint metadata.

The `mutag` experiment name denotes the 4,337-graph Mutagenicity dataset, not the
188-graph MUTAG dataset. BA-2Motif's topology-aware reference uses motif-edge
supervision and is evaluated separately from generic recursive CIDER.

From the repository root:

```bash
sha256sum -c data/SHA256SUMS
cd cider_minimal
python show_results.py --recompute --split test
```

Re-randomizing splits or regenerating the synthetic graphs will not reproduce
the saved metrics. For new studies, prepare independent splits and select
hyperparameters only on validation data. See the root and code-package READMEs
for model settings and the distinction between fidelity and label accuracy.
