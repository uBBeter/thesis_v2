# Thesis Project Context

## Overview

**Topic**: Graph Neural Networks for Recommendations in E-Commerce  
**Type**: Bachelor Thesis  
**Goal**: Compare several GNN architectures against a baseline (ALS) on e-commerce datasets, measuring recommendation quality metrics.

---

## Datasets

**Source**: Amazon Reviews 2023 (McAuley Lab, UCSD)

| Dataset | Category | File |
|---------|----------|------|
| Amazon Beauty | Beauty products | `data/raw/beauty/All_Beauty.jsonl.gz` |
| Amazon CDs & Vinyl | Music | `data/raw/cds/CDs_and_Vinyl.jsonl.gz` |

**Preprocessing pipeline**:
1. Parse JSONL.GZ → DataFrame (`user`, `item`, `rating`, `timestamp`)
2. K-core filtering (k=10): remove users/items with < 10 interactions
3. ID remapping: string IDs → contiguous integers
4. Leave-one-out split: last interaction = test, second-to-last = val, rest = train; 99 negative samples per user
5. Graph construction: symmetric bipartite user-item graph with D^{-1/2} A D^{-1/2} normalization

Output: `data/processed/{dataset}/dataset.pkl` + `graph.pt`

---

## Models

### GNN Architectures

| Model | Paper | Key Idea |
|-------|-------|----------|
| **LightGCN** | He et al., 2020 | Simplified GCN — neighborhood aggregation + mean pooling across layers, BPR loss |
| **NGCF** | Wang et al., 2019 | Neural graph CF — feature transformation (W1, W2), LeakyReLU, concatenates layer embeddings |
| **SGL** | Wu et al., 2021 | LightGCN + self-supervised contrastive learning via edge/node dropout augmentation, InfoNCE loss |
| **SimGCL** | Yu et al., 2022 | Simpler SGL — adds uniform noise to embeddings instead of graph augmentation, InfoNCE loss |

### Baseline

| Model | Description |
|-------|-------------|
| **ALS** | Alternating Least Squares via `implicit` library — non-graph matrix factorization |

### Common Training Details
- Loss: BPR (Bayesian Personalized Ranking) for all GNN models
- Optimizer: Adam
- Validation metric for early stopping: Recall@20
- Default: 64-dim embeddings, 3 GNN layers, lr=1e-3, batch_size=2048, epochs=200, patience=10

---

## Evaluation

**Protocol**: Leave-one-out — each user has 1 positive + 99 negatives at test/val time  
**Metrics**: Recall@10, Recall@20, NDCG@10, NDCG@20, Precision@10, Precision@20  
**Results saved to**: `results/{dataset}/{model}/metrics.json`

---

## Project Structure

```
thesis_v2/
├── configs/              # YAML configs for each model
│   ├── lightgcn.yaml
│   ├── ngcf.yaml
│   ├── sgl.yaml
│   ├── simgcl.yaml
│   └── als.yaml
├── data/
│   ├── raw/              # Downloaded JSONL.GZ files
│   └── processed/        # dataset.pkl + graph.pt
├── results/              # metrics.json per model/dataset
├── scripts/
│   ├── download_data.py  # Download datasets
│   ├── preprocess.py     # Preprocess raw data
│   ├── train.py          # Main training entry point
│   └── evaluate.py       # Load and display saved metrics
├── src/
│   ├── data/             # loader.py, preprocessor.py, graph.py
│   ├── models/           # base.py, lightgcn.py, ngcf.py, sgl.py, simgcl.py, als.py
│   ├── evaluation/       # metrics.py
│   └── utils/
├── requirements.txt
└── setup.sh              # One-command install for GPU instances
```

---

## Quick Start

```bash
bash setup.sh
python scripts/download_data.py --dataset all
python scripts/preprocess.py --dataset all

# Train (run on GPU instance)
python scripts/train.py --config configs/lightgcn.yaml --dataset beauty --device cuda
python scripts/train.py --config configs/ngcf.yaml --dataset beauty --device cuda
python scripts/train.py --config configs/sgl.yaml --dataset beauty --device cuda
python scripts/train.py --config configs/simgcl.yaml --dataset beauty --device cuda
python scripts/train.py --config configs/als.yaml --dataset beauty --device cpu

# Evaluate
python scripts/evaluate.py --all --dataset beauty
```

---

## Status & Known Issues

- Code scaffold is complete: all 5 models, both datasets, full training/eval pipeline
- Has been deployed to a remote GPU instance for training
- **Issues encountered when running on remote GPU** — details TBD (context was lost when Claude session restarted)
- Next step: debug and resolve GPU training issues

---

## Architecture Notes

- All models inherit from `BaseRecommender` (ABC) with `fit()`, `get_user_embeddings()`, `get_item_embeddings()`
- Bipartite graph: user nodes 0..n_users-1, item nodes n_users..n_users+n_items-1
- `setup.sh` auto-detects CUDA version and installs matching torch-geometric + sparse ops
