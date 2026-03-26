#!/bin/bash -l
#SBATCH --job-name=sweep
#SBATCH --partition=rali
#SBATCH --gres=gpu:rtx_a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=120
#SBATCH --output=logs/slurm/%j.out

SWEEP_ID="${1:?Usage: sbatch scripts/sweep.sh <sweep_id>}"

PROJECT_DIR="$HOME/nlp-benchmark-taxonomy"
SCRATCH_DIR="/Tmp/$(whoami)/${SLURM_JOB_ID}"

# Copy local project
mkdir -p "$SCRATCH_DIR/data/splits/ready"
mkdir -p "$SCRATCH_DIR/data/models"
cp -r "$PROJECT_DIR"/{src,config,requirements-train.txt,pyproject.toml,uv.lock} "$SCRATCH_DIR/"
cp -r "$PROJECT_DIR/data/splits/ready/"* "$SCRATCH_DIR/data/splits/ready/"

cd "$SCRATCH_DIR"

# Setup env
VENV_DIR="$PROJECT_DIR/.venv"
module load python/3.11 2>/dev/null || true
REQ_HASH=$(md5sum "$SCRATCH_DIR/requirements-train.txt" | cut -d' ' -f1)
if [ ! -f "$VENV_DIR/.req_hash" ] || [ "$REQ_HASH" != "$(cat "$VENV_DIR/.req_hash")" ]; then
    echo "Requirements changed — reinstalling..."
    python -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --quiet -r requirements-train.txt
    echo "$REQ_HASH" > "$VENV_DIR/.req_hash"
else
    echo "Reusing cached venv."
    source "$VENV_DIR/bin/activate"
fi

export PYTHONPATH="$SCRATCH_DIR/src:$PYTHONPATH"

# Run sweep agent — picks up runs until sweep is done or job times out
wandb agent "$SWEEP_ID"

# Copy trained models back to persistent storage
cp -r "$SCRATCH_DIR/data/models/"* "$PROJECT_DIR/data/models/" 2>/dev/null || true
