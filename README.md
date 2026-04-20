# GNN-Based Recommendation Systems for E-Commerce

Bachelor thesis comparing Graph Neural Network architectures against a matrix factorization baseline (ALS) on Amazon e-commerce datasets.

## Models

| Model | Description |
|-------|-------------|
| **LightGCN** | Simplified GCN — pure neighborhood aggregation, mean pooling across layers |
| **NGCF** | Neural graph CF — adds feature transformation and non-linear activation |
| **SGL** | LightGCN + self-supervised contrastive learning via graph augmentation |
| **SimGCL** | SGL variant — replaces graph augmentation with uniform embedding noise |
| **ALS** | Alternating Least Squares baseline (matrix factorization, no graph) |

## Datasets

| Dataset | Category |
|---------|----------|
| Amazon Toys & Games | Toy and game products |
| Amazon CDs & Vinyl | Music |

Both datasets are from the [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) collection and preprocessed with k=10 core filtering and leave-one-out splits.

## Results

### Toys & Games

| Model | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 | Precision@10 | Precision@20 |
|-------|-----------|-----------|---------|---------|--------------|--------------|
| **LightGCN** | **0.4687** | **0.6386** | **0.2847** | **0.3274** | **0.0469** | **0.0319** |
| NGCF | 0.4507 | 0.6245 | 0.2649 | 0.3087 | 0.0451 | 0.0312 |
| SGL | 0.4367 | 0.5882 | 0.2730 | 0.3111 | 0.0437 | 0.0294 |
| SimGCL | 0.3436 | 0.4652 | 0.2204 | 0.2510 | 0.0344 | 0.0233 |
| ALS | 0.3467 | 0.4197 | 0.3080 | 0.3262 | 0.0347 | 0.0210 |

### CDs & Vinyl

| Model | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 | Precision@10 | Precision@20 |
|-------|-----------|-----------|---------|---------|--------------|--------------|
| **LightGCN** | **0.6930** | **0.8251** | 0.4539 | 0.4874 | **0.0693** | **0.0413** |
| NGCF | 0.6522 | 0.7942 | 0.4143 | 0.4504 | 0.0652 | 0.0397 |
| SGL | 0.6867 | 0.8100 | **0.4585** | **0.4898** | 0.0687 | 0.0405 |
| SimGCL | 0.5754 | 0.6898 | 0.3983 | 0.4272 | 0.0575 | 0.0345 |
| ALS | 0.1887 | 0.2787 | 0.1403 | 0.1627 | 0.0188 | 0.0139 |

---

## Setup

```bash
pip install -r requirements.txt
```

For GPU training (CUDA instance), use the setup script instead — it auto-detects your CUDA version and installs matching torch-geometric:

```bash
bash setup.sh
```

---

## Quickstart: View Results Locally

Pre-trained weights and metrics are available on HuggingFace Hub at [ubbeter/thesis-gnn-recsys](https://huggingface.co/ubbeter/thesis-gnn-recsys). To download weights and see recommendations without training anything:

```bash
# Download weights and show top-10 recommendations for 5 random users
python scripts/demo.py \
    --model lightgcn \
    --dataset toys \
    --config configs/lightgcn.yaml \
    --hf-repo ubbeter/thesis-gnn-recsys

# Try other models
python scripts/demo.py --model ngcf    --dataset toys --config configs/ngcf.yaml    --hf-repo ubbeter/thesis-gnn-recsys
python scripts/demo.py --model sgl     --dataset toys --config configs/sgl.yaml     --hf-repo ubbeter/thesis-gnn-recsys
python scripts/demo.py --model simgcl  --dataset toys --config configs/simgcl.yaml  --hf-repo ubbeter/thesis-gnn-recsys
```

To print the full results table for a dataset:

```bash
python scripts/evaluate.py --all --dataset toys
python scripts/evaluate.py --all --dataset cds
```

---

## Full Reproduction

### 1. Download and preprocess data

```bash
python scripts/download_data.py --dataset all
python scripts/preprocess.py --dataset all
```

### 2. Train models

Requires a CUDA GPU. Run on a GPU instance:

```bash
# Toys & Games
python scripts/train.py --config configs/lightgcn.yaml --dataset toys --device cuda
python scripts/train.py --config configs/ngcf.yaml     --dataset toys --device cuda
python scripts/train.py --config configs/sgl.yaml      --dataset toys --device cuda
python scripts/train.py --config configs/simgcl.yaml   --dataset toys --device cuda
python scripts/train.py --config configs/als.yaml      --dataset toys --device cpu

# CDs & Vinyl
python scripts/train.py --config configs/lightgcn.yaml --dataset cds --device cuda
python scripts/train.py --config configs/ngcf.yaml     --dataset cds --device cuda
python scripts/train.py --config configs/sgl.yaml      --dataset cds --device cuda
python scripts/train.py --config configs/simgcl.yaml   --dataset cds --device cuda
python scripts/train.py --config configs/als.yaml      --dataset cds --device cpu
```

Checkpoints are saved to `results/{dataset}/{model}/best.pt`. Training uses early stopping on Recall@20 with patience=10 (evaluated every 5 epochs).

### 3. Evaluate

```bash
python scripts/evaluate.py --all --dataset toys
python scripts/evaluate.py --all --dataset cds
```

---

## Project Structure

```
thesis_v2/
├── configs/              # YAML hyperparameter configs per model
├── data/
│   ├── raw/              # Downloaded JSONL.GZ files (gitignored)
│   └── processed/        # dataset.pkl + graph.pt (gitignored)
├── results/              # metrics.json + best.pt checkpoints per model/dataset
├── scripts/
│   ├── download_data.py  # Download Amazon datasets
│   ├── preprocess.py     # Preprocess raw data
│   ├── train.py          # Train a single model
│   ├── evaluate.py       # Print results table from saved metrics
│   ├── demo.py           # Load model and show recommendations
│   └── upload_weights.py # Upload checkpoints to HuggingFace Hub
├── src/
│   ├── data/             # loader.py, preprocessor.py, graph.py
│   ├── models/           # base.py, lightgcn.py, ngcf.py, sgl.py, simgcl.py, als.py
│   ├── evaluation/       # metrics.py
│   └── utils/            # helpers.py
├── requirements.txt
└── setup.sh
```

---

## Evaluation Protocol

- **Split**: Leave-one-out — last interaction per user = test, second-to-last = validation
- **Negatives**: 99 randomly sampled negative items per user at test/val time
- **Metrics**: Recall@10, Recall@20, NDCG@10, NDCG@20, Precision@10, Precision@20
- **Early stopping**: based on Recall@20 on the validation set
