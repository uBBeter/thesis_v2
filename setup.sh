#!/bin/bash
# Run this once on a fresh GPU instance to install everything.
# Usage: bash setup.sh
set -e

echo "=== Detecting PyTorch / CUDA ==="
TORCH=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA=$(python -c "import torch; v=torch.version.cuda; print('cu'+''.join(v.split('.')[:2])) if v else print('cpu')" 2>/dev/null || echo "cpu")
echo "PyTorch: $TORCH  |  CUDA: $CUDA"

echo ""
echo "=== Installing torch-geometric ==="
pip install torch-geometric --quiet

echo ""
echo "=== Installing optional PyG extensions (faster sparse ops) ==="
pip install pyg_lib torch_scatter torch_sparse \
    -f "https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html" --quiet 2>/dev/null \
    || echo "Optional PyG extensions skipped (not required for these models)"

echo ""
echo "=== Installing remaining dependencies ==="
pip install implicit pandas numpy scipy scikit-learn \
            pyyaml matplotlib seaborn tqdm requests --quiet

echo ""
echo "=== Done! Next steps ==="
echo "  python scripts/download_data.py --dataset all"
echo "  python scripts/preprocess.py --dataset all"
echo "  python scripts/train.py --config configs/lightgcn.yaml --dataset beauty --device cuda"
