#!/usr/bin/env bash
set -euo pipefail

# Reproduce Scandium Labs training and evaluation pipeline
# Usage: bash reproduce.sh [dataset_path] [config_path]

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

DATASET_PATH="${1:-datasets/v3_li_10000}"
CONFIG_PATH="${2:-configs/model_config_v3_li.yaml}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/reproduce_${TIMESTAMP}"

echo "=== Scandium Labs Reproducibility Run ==="
echo "Repository: $REPO_DIR"
echo "Dataset:    $DATASET_PATH"
echo "Config:     $CONFIG_PATH"
echo "Output:     $OUTPUT_DIR"
echo "Date:       $(date)"
echo ""

# Step 1: Environment check
echo "=== Step 1: Checking environment ==="
python --version 2>&1
pip list 2>/dev/null | grep -i "torch\|pytorch_geometric\|scandium" || true
echo ""

# Step 2: Install package
echo "=== Step 2: Installing package ==="
pip install -e . -q
echo ""

# Step 3: Verify dataset
echo "=== Step 3: Verifying dataset ==="
if [ -d "$DATASET_PATH" ]; then
    echo "Dataset found at $DATASET_PATH"
    ls -la "$DATASET_PATH" | head -10
else
    echo "WARNING: Dataset not found at $DATASET_PATH"
    echo "Run scripts/preprocess/build_dataset.py first."
fi
echo ""

# Step 4: Run tests
echo "=== Step 4: Running tests ==="
python -m pytest tests/ -q --tb=short -k "not test_reference_materials and not test_training_normalization" 2>&1 | tail -5
echo ""

# Step 5: Train model
echo "=== Step 5: Training model ==="
mkdir -p "$OUTPUT_DIR/checkpoints"
python -c "
from src.training import ScandiumTrainer
trainer = ScandiumTrainer('$CONFIG_PATH', data_dir='$DATASET_PATH')
model, metrics = trainer.train()
print(f'Training complete. Final metrics: {metrics}')
import json
with open('$OUTPUT_DIR/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
" 2>&1 | tee "$OUTPUT_DIR/training.log"
echo ""

# Step 6: Evaluate
echo "=== Step 6: Evaluating model ==="
python scripts/evaluate/cross_validate.py --config "$CONFIG_PATH" --output "$OUTPUT_DIR/evaluation" 2>&1 | tee "$OUTPUT_DIR/evaluation.log"
echo ""

# Step 7: Summary
echo "=== Reproduction Summary ==="
echo "Output directory: $OUTPUT_DIR"
echo "Metrics:          $OUTPUT_DIR/metrics.json"
echo "Evaluation:       $OUTPUT_DIR/evaluation"
echo "Logs:             $OUTPUT_DIR/training.log"
echo ""
echo "=== Reproduction complete ==="
