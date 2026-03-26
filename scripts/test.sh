#!/bin/bash -l
#SBATCH --job-name=test
#SBATCH --partition=rali
#SBATCH --gres=gpu:rtx_a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=10
#SBATCH --output=logs/slurm/%j.out

CHECKPOINT="${1:?Usage: sbatch scripts/test.sh <checkpoint_path> <config>}"
CONFIG="${2:?Usage: sbatch scripts/test.sh <checkpoint_path> <config>}"

PROJECT_DIR="$HOME/nlp-benchmark-taxonomy"
SCRATCH_DIR="/Tmp/$(whoami)/${SLURM_JOB_ID}"

# Copy project + only the needed checkpoint
mkdir -p "$SCRATCH_DIR/data/splits/ready"
cp -r "$PROJECT_DIR"/{src,config,requirements-train.txt,pyproject.toml,uv.lock} "$SCRATCH_DIR/"
cp -r "$PROJECT_DIR/data/splits/ready/"* "$SCRATCH_DIR/data/splits/ready/"
mkdir -p "$SCRATCH_DIR/$CHECKPOINT"
if [ -d "$SCRATCH_DIR/$CHECKPOINT" ] && [ -f "$SCRATCH_DIR/$CHECKPOINT/model.safetensors" ]; then
    echo "Checkpoint already on scratch, skipping copy."
else
    echo "Copying checkpoint to scratch."
    cp -r "$PROJECT_DIR/$CHECKPOINT/"* "$SCRATCH_DIR/$CHECKPOINT/"
fi

cd "$SCRATCH_DIR"

# Setup env
VENV_DIR="$PROJECT_DIR/.venv"
module load python/3.11 2>/dev/null || true
source "$VENV_DIR/bin/activate"

export PYTHONPATH="$SCRATCH_DIR/src:$PYTHONPATH"
python -m src.training.trainer --test-only "$CHECKPOINT" --config "$CONFIG"

# Copy predictions back
cp -r "$SCRATCH_DIR/$CHECKPOINT/test_predictions.parquet" "$PROJECT_DIR/$CHECKPOINT/" 2>/dev/null || true

echo "Done."
