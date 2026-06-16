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

## Datasets Used in the Paper

The paper evaluates ETDTGN on 13 public continuous-time dynamic graph (CTDG) datasets. The dataset list and chronological evaluation protocol follow recent temporal graph benchmark practice [Huang2023, Shirzadkhani2024].

- **Wikipedia**: A bipartite user-page editing network. Each temporal edge denotes a user editing a Wikipedia page and includes text-derived edge features. Dynamic user labels are available for node classification [Kumar2019, Rossi2020, Huang2023].
- **Reddit**: A bipartite user-subreddit interaction network. Each temporal edge records a user posting to a subreddit and includes text-derived edge features. Dynamic user labels are available for node classification [Kumar2019, Rossi2020, Huang2023].
- **MOOC**: A bipartite student-course-content interaction network. Temporal edges record student actions on online course materials such as videos or exercises [Kumar2019, Huang2023].
- **LastFM**: A user-song listening network where timestamped edges represent music listening events [Kumar2019, Huang2023].
- **Enron**: An email communication network among Enron employees, where temporal edges correspond to email interactions [Huang2023, Shirzadkhani2024].
- **Social Evo.**: A proximity/social interaction dataset used to evaluate temporal models on dense human-contact dynamics [Huang2023, Shirzadkhani2024].
- **UCI**: An online communication network among university users, where temporal edges correspond to message interactions [Huang2023, Shirzadkhani2024].
- **Flights**: A transport network where temporal edges represent flight connections or time-stamped mobility interactions [Huang2023, Shirzadkhani2024].
- **Can. Parl.**: A Canadian Parliament interaction network for political temporal-graph evaluation [Huang2023, Shirzadkhani2024].
- **US Legis.**: A U.S. legislation interaction network that captures temporal political/co-sponsorship or legislative relationships [Huang2023, Shirzadkhani2024].
- **UN Trade**: A temporal international trade network used to evaluate dynamic interaction modeling in economic graphs [Huang2023, Shirzadkhani2024].
- **UN Vote**: A temporal voting-agreement network among countries in United Nations voting records [Huang2023, Shirzadkhani2024].
- **Contact**: A temporal proximity/contact network used to test dynamic link prediction under human-contact patterns [Huang2023, Shirzadkhani2024].

In the paper, these datasets are grouped by domain: social (`Wikipedia`, `Reddit`, `Enron`, `UCI`), interaction (`MOOC`, `LastFM`), proximity (`Social Evo.`, `Contact`), transport (`Flights`), politics (`Can. Parl.`, `US Legis.`, `UN Vote`), and economics (`UN Trade`). Dynamic link prediction is evaluated with chronological train/validation/test splits, and dynamic node classification is evaluated on datasets with temporal node labels.

## Baselines Used in the Paper

The paper compares ETDTGN with 15 representative CTDG baselines covering memory-based models, attention/history encoders, non-parametric memory baselines, long-range propagation models, and scalable temporal-graph systems.

- **JODIE**: A recurrent temporal interaction model that maintains coupled dynamic embeddings and projects them to future timestamps [Kumar2019].
- **DyRep**: A memory-based dynamic graph model that updates node states after events and models temporal interaction intensity [Trivedi2019].
- **TGAT**: A temporal graph attention model that combines temporal-neighbor aggregation with functional time encoding [Xu2020].
- **TGN**: A modular memory-based CTDG framework with message construction, message aggregation, memory update, and temporal embedding modules [Rossi2020].
- **CAWN**: A causal anonymous-walk model for inductive temporal network representation learning [Wang2021CAWN].
- **EdgeBank**: A non-parametric memory baseline for dynamic link prediction that predicts links from stored historical edges [Poursafaei2022].
- **TCL**: A Transformer-based dynamic graph model with contrastive learning for temporal representation learning [Wang2021TCL].
- **GraphMixer**: A lightweight temporal graph model showing that MLP/token-mixing designs can be competitive with more complex temporal architectures [Cong2023].
- **DyGFormer**: A patch-based Transformer architecture for dynamic graph learning that encodes source/destination interaction histories and neighbor co-occurrence features [Yu2023].
- **CTAN**: A long-range propagation model for continuous-time dynamic graphs [Gravina2024].
- **TGNv2**: An enhanced TGN variant that improves message expressivity by incorporating source-target identification [Tjandra2024].
- **DyGMamba**: A state-space-model-based CTDG method for efficiently modeling long-term temporal dependencies [Ding2024].
- **TIGER**: A restart-based temporal interaction graph embedding framework designed to improve parallel training and scalability [Zhang2023].
- **PRES**: A scalable memory-based dynamic GNN framework that addresses temporal discontinuity with prediction-correction and memory-coherence learning [Su2024].
- **TGL**: A general temporal GNN training framework for large-scale graphs with temporal sampling and memory-oriented components [Zhou2022].

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

## References

- [Rossi2020] E. Rossi, B. Chamberlain, F. Frasca, D. Eynard, F. Monti, and M. Bronstein. Temporal Graph Networks for Deep Learning on Dynamic Graphs. ICML, 2020.
- [Kumar2019] S. Kumar, X. Zhang, and J. Leskovec. Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks. KDD, 2019.
- [Trivedi2019] R. Trivedi, M. Farajtabar, P. Biswal, and H. Zha. DyRep: Learning Representations over Dynamic Graphs. ICLR, 2019.
- [Xu2020] D. Xu, C. Ruan, E. Korpeoglu, S. Kumar, and K. Achan. Inductive Representation Learning on Temporal Graphs. ICLR, 2020.
- [Wang2021CAWN] Y. Wang, Y.-Y. Chang, Y. Liu, J. Leskovec, and P. Li. Inductive Representation Learning in Temporal Networks via Causal Anonymous Walks. ICLR, 2021.
- [Poursafaei2022] F. Poursafaei, A. Huang, K. Pelrine, and R. Rabbany. Towards Better Evaluation for Dynamic Link Prediction. NeurIPS Datasets and Benchmarks, 2022.
- [Wang2021TCL] L. Wang et al. TCL: Transformer-based Dynamic Graph Modelling via Contrastive Learning. arXiv, 2021.
- [Cong2023] W. Cong, S. Zhang, J. Kang, B. Yuan, H. Wu, X. Zhou, H. Tong, and M. Mahdavi. Do We Really Need Complicated Model Architectures For Temporal Networks? ICLR, 2023.
- [Yu2023] L. Yu, L. Sun, B. Du, and W. Lv. Towards Better Dynamic Graph Learning: New Architecture and Unified Library. NeurIPS, 2023.
- [Gravina2024] A. Gravina, G. Lovisotto, G. Gallicchio, D. Bacciu, and C. Grohnfeldt. Long Range Propagation on Continuous-Time Dynamic Graphs. ICML, 2024.
- [Tjandra2024] B. A. Tjandra, F. Barbero, and M. Bronstein. Enhancing the Expressivity of Temporal Graph Networks through Source-Target Identification. NeurIPS, 2024.
- [Ding2024] Z. Ding et al. DyGMamba: Efficiently Modeling Long-Term Temporal Dependency on Continuous-Time Dynamic Graphs with State Space Models. TMLR, 2024.
- [Zhang2023] Y. Zhang et al. TIGER: Temporal Interaction Graph Embedding with Restarts. WWW, 2023.
- [Su2024] J. Su, D. Zou, and C. Wu. PRES: Toward Scalable Memory-Based Dynamic Graph Neural Networks. ICLR, 2024.
- [Zhou2022] H. Zhou, D. Zheng, I. Nisa, V. Ioannidis, X. Song, and G. Karypis. TGL: A General Framework for Temporal GNN Training on Billion-Scale Graphs. VLDB, 2022.
- [Huang2023] S. Huang et al. Temporal Graph Benchmark for Machine Learning on Temporal Graphs. NeurIPS, 2023.
- [Shirzadkhani2024] R. Shirzadkhani, S. Huang, E. Kooshafar, R. Rabbany, and F. Poursafaei. Temporal Graph Analysis with TGX. WSDM, 2024.
