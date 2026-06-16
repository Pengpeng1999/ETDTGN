# ETDTGN: Event-Type Decomposition Temporal Graph Network

ETDTGN is a memory-based continuous-time dynamic graph model built on the TGN pipeline. The main change is the `type_based` message aggregator: concurrent messages of the same node are projected into a learnable event-type space, weighted by type-specific attention, and then aggregated before memory update. This is intended to reduce semantic mixing and salience dilution caused by simple `last` or `mean` aggregation.

## Main Components

- `train_self_supervised.py`: link prediction training and evaluation.
- `train_supervised.py`: node classification evaluation using a trained temporal encoder.
- `modules/message_aggregator.py`: implements `last`, `mean`, and ETDTGN's `type_based` aggregator.
- `model/tgn.py`: TGN-style memory, message, embedding, and prediction pipeline used as the backbone.
- `utils/data_processing.py`: dataset loading and temporal train/validation/test split.
- `evaluation/evaluation.py`: AP/AUC evaluation for link prediction and AUC evaluation for node classification.

## Data Format

Datasets are expected under `data/` with the preprocessed TGN-style files:

- `ml_<dataset>.csv`
- `ml_<dataset>.npy`
- `ml_<dataset>_node.npy`

The loader accepts dataset names such as `wikipedia`, `reddit`, `mooc`, and `lastfm` when the corresponding files exist.

## Link Prediction

Default ETDTGN setting:

```bash
python train_self_supervised.py -d wikipedia --use_memory --aggregator type_based --n_types 2 --prefix etdtgn-wikipedia
```

Run on another dataset:

```bash
python train_self_supervised.py -d reddit --use_memory --aggregator type_based --n_types 2 --prefix etdtgn-reddit
```

Ablation baselines:

```bash
python train_self_supervised.py -d wikipedia --use_memory --aggregator mean --prefix tgn-mean-wikipedia
python train_self_supervised.py -d wikipedia --use_memory --aggregator last --prefix tgn-last-wikipedia
```

Sensitivity to the number of event types:

```bash
python train_self_supervised.py -d wikipedia --use_memory --aggregator type_based --n_types 3 --prefix etdtgn-k3-wikipedia
python train_self_supervised.py -d wikipedia --use_memory --aggregator type_based --n_types 4 --prefix etdtgn-k4-wikipedia
```

## Node Classification

First train or load a self-supervised ETDTGN encoder, then train the node-classification decoder:

```bash
python train_self_supervised.py -d wikipedia --use_memory --aggregator type_based --n_types 2 --prefix etdtgn-wikipedia
python train_supervised.py -d wikipedia --use_memory --aggregator type_based --n_types 2 --prefix etdtgn-wikipedia
```

## Important Arguments

- `--aggregator {type_based,mean,last}`: message aggregation strategy. `type_based` is ETDTGN's default.
- `--n_types`: number of learnable event-type bases used by `type_based`.
- `--use_memory`: enables node memory.
- `--n_degree`: number of temporal neighbors sampled by the embedding module.
- `--bs`: temporal batch size.
- `--n_epoch`: number of training epochs.
- `--prefix`: prefix for checkpoints, saved models, and result files.

## Outputs

- `saved_models/`: final model checkpoints.
- `saved_checkpoints/`: epoch-level checkpoints for early stopping.
- `results/`: pickled metrics including validation AP, test AP, and epoch time.
- `log/`: training logs.

## Notes

This repository keeps the TGN modular backbone because ETDTGN modifies the message aggregation interface rather than replacing the whole continuous-time dynamic graph pipeline. Therefore, files such as `model/tgn.py`, `modules/memory.py`, and temporal embedding modules are still part of the method implementation.

The code is intended for reproducing ETDTGN experiments and ablations. Generated caches, old project-introduction materials, and unrelated TGN presentation files are not required for running the experiments.
