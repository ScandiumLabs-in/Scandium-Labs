#!/bin/bash
# Autopilot: monitors training, runs evaluation + benchmark + comparison when done
set -e

EXPERIMENT="v2_3635_first_run"
LOG_FILE="experiments/${EXPERIMENT}/train.log"
LAST_CHECK=""
IDLE_THRESHOLD=600  # 10 min without checkpoint update = done

echo "[Autopilot] Monitoring training for ${EXPERIMENT}..."
echo "[Autopilot] Started at $(date)"

while true; do
    # Find latest checkpoint
    LATEST=$(ls -1t checkpoints/epoch_*.pt 2>/dev/null | head -1)
    
    if [ -n "$LATEST" ]; then
        CURRENT_TIME=$(stat -c %Y "$LATEST")
        
        if [ "$CURRENT_TIME" != "$LAST_CHECK" ]; then
            EPOCH=$(basename "$LATEST" | sed 's/epoch_//;s/\.pt//')
            NOW=$(date +%s)
            AGE=$(( (NOW - CURRENT_TIME) / 60 ))
            echo "[Autopilot] $(date): Epoch ${EPOCH} checkpoint saved ${AGE} min ago"
            LAST_CHECK=$CURRENT_TIME
        fi
    fi
    
    # Check if process is still running
    PID=$(pgrep -f "run_experiment.*${EXPERIMENT}" 2>/dev/null | head -1)
    
    if [ -z "$PID" ]; then
        echo "[Autopilot] $(date): Training process not found, checking if complete..."
        
        # Check if we have a recent checkpoint (within IDLE_THRESHOLD)
        LATEST_BEST=$(stat -c %Y checkpoints/best_model.pt 2>/dev/null || echo 0)
        NOW=$(date +%s)
        AGE=$(( NOW - LATEST_BEST ))
        
        if [ $AGE -lt $IDLE_THRESHOLD ] && [ -f "checkpoints/best_model.pt" ]; then
            echo "[Autopilot] Training appears complete (best_model.pt saved ${AGE}s ago)"
            break
        else
            echo "[Autopilot] No recent activity. Waiting..."
            sleep 60
            continue
        fi
    fi
    
    sleep 60
done

echo ""
echo "[Autopilot] === TRAINING COMPLETE ==="
echo "[Autopilot] Time: $(date)"

# Step 1: Copy experiment artifacts
echo "[Autopilot] Copying experiment artifacts..."
cp checkpoints/best_model.pt "experiments/${EXPERIMENT}/checkpoint.pt"
cp checkpoints/*.pt "experiments/${EXPERIMENT}/" 2>/dev/null || true

# Step 2: Generate training plot
echo "[Autopilot] Generating training plot..."
cd /home/shamique/Scandium\ Labs\ SSB/scandium-labs
"/home/shamique/Scandium Labs SSB/scandium-labs/venv/bin/python" -c "
import json, re, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Parse logs for loss values
log_file = 'experiments/${EXPERIMENT}/train.log'
losses = {'train': [], 'val': []}
with open(log_file) as f:
    for line in f:
        m = re.search(r'epoch: (\d+).*train_loss: ([\d.]+).*val_loss: ([\d.]+)', line)
        if m:
            losses['train'].append(float(m.group(2)))
            losses['val'].append(float(m.group(3)))

# Try checkpoint metrics
import torch
ckpt = torch.load('checkpoints/best_model.pt', weights_only=False)
metrics = ckpt.get('metrics', {})

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ax1, ax2 = axes

if losses['train']:
    ax1.plot(losses['train'], label='Train', color='black')
    ax1.plot(losses['val'], label='Val', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Training Loss')

# Metrics bar chart
task_names = []
mae_values = []
for k, v in sorted(metrics.items()):
    if k.endswith('_mae'):
        task_names.append(k.replace('_mae', ''))
        mae_values.append(v)

if task_names:
    ax2.barh(task_names, mae_values, color='black')
    ax2.set_xlabel('MAE')
    ax2.set_title('Test Metrics')

plt.tight_layout()
plt.savefig('experiments/${EXPERIMENT}/training_curves.png', dpi=150, bbox_inches='tight')
print('[Autopilot] Training curves saved')
" 2>&1

# Step 3: Run benchmark suite
echo "[Autopilot] Running benchmark suite..."
"/home/shamique/Scandium Labs SSB/scandium-labs/venv/bin/python" -W ignore scripts/benchmark_suite.py 2>&1 | tee "experiments/${EXPERIMENT}/benchmark.log"

# Step 4: Run cross-validation
echo "[Autopilot] Running 5-fold cross-validation..."
"/home/shamique/Scandium Labs SSB/scandium-labs/venv/bin/python" -W ignore scripts/cross_validate.py \
    --config configs/model_config_v2.yaml \
    --data_dir datasets/v2_10000 2>&1 | tee "experiments/${EXPERIMENT}/cv.log"

# Step 5: Generate comparison table
echo "[Autopilot] Generating comparison..."
"/home/shamique/Scandium Labs SSB/scandium-labs/venv/bin/python" -c "
import json, torch

# Load v1 baseline
v1 = json.load(open('data/baseline_v1.0.json'))
v1_metrics = v1.get('test_metrics', {})
v1_ef_mae = v1_metrics.get('formation_energy_mae', '?')
v1_ef_r2 = v1_metrics.get('formation_energy_r2', '?')
v1_bg_mae = v1_metrics.get('band_gap_mae', '?')
v1_bg_r2 = v1_metrics.get('band_gap_r2', '?')
v1_eah_r2 = '?'  # Not stored in baseline

# Load CV results if available
try:
    cv = json.load(open('experiments/cv/summary.json'))
    v2_ef_mae = cv['aggregate']['formation_energy']['mae']['mean']
    v2_ef_r2 = cv['aggregate']['formation_energy']['r2']['mean']
    v2_bg_mae = cv['aggregate']['band_gap']['mae']['mean']
    v2_bg_r2 = cv['aggregate']['band_gap']['r2']['mean']
    v2_eah_r2 = cv['aggregate']['energy_above_hull']['r2']['mean']
except Exception:
    v2_ef_mae = v2_ef_r2 = v2_bg_mae = v2_bg_r2 = v2_eah_r2 = '?'

print()
print('=' * 70)
print('  v1 vs v2 COMPARISON')
print('=' * 70)
print(f'  {\"Metric\":30s} {\"v1 (817)\":>12s} {\"v2 (3635)\":>12s} {\"Δ\":>8s}')
print('  ' + '-' * 62)
print(f'  {\"Formation Energy MAE\":30s} {str(v1_ef_mae):>12s} {str(v2_ef_mae):>12s} {\"?\":>8s}')
print(f'  {\"Formation Energy R²\":30s} {str(v1_ef_r2):>12s} {str(v2_ef_r2):>12s} {\"?\":>8s}')
print(f'  {\"Band Gap MAE\":30s} {str(v1_bg_mae):>12s} {str(v2_bg_mae):>12s} {\"?\":>8s}')
print(f'  {\"Band Gap R²\":30s} {str(v1_bg_r2):>12s} {str(v2_bg_r2):>12s} {\"?\":>8s}')
print(f'  {\"Energy Above Hull R²\":30s} {str(v1_eah_r2):>12s} {str(v2_eah_r2):>12s} {\"?\":>8s}')
print('=' * 70)
print()
" 2>&1 | tee "experiments/${EXPERIMENT}/comparison.txt"

echo ""
echo "[Autopilot] === ALL DONE ==="
echo "[Autopilot] Results in experiments/${EXPERIMENT}/"
echo "[Autopilot] Time: $(date)"
