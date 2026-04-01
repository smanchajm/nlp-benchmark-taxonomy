#!/bin/bash -l
#SBATCH --job-name=train-0
#SBATCH --partition=rali
#SBATCH --gres=gpu:rtx_a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=60
#SBATCH --output=logs/slurm/%j.out

# Out dir
PROJECT_DIR="$HOME/nlp-benchmark-taxonomy"
SCRATCH_DIR="/Tmp/$(whoami)/${SLURM_JOB_ID}"

cleanup() { rm -rf "$SCRATCH_DIR"; echo "Cleaned up $SCRATCH_DIR"; }
trap cleanup EXIT

# Copy local project
mkdir -p "$SCRATCH_DIR/data/splits/ready"
mkdir -p "$SCRATCH_DIR/data/models"
cp -r "$PROJECT_DIR"/{src,config,requirements-train.txt,pyproject.toml,uv.lock} "$SCRATCH_DIR/"
cp -r "$PROJECT_DIR/data/splits/ready/"* "$SCRATCH_DIR/data/splits/ready/"

cd "$SCRATCH_DIR"

# Setup env (persistent venv, only reinstall if requirements change)
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

# Train
export PYTHONPATH="$SCRATCH_DIR/src:$PYTHONPATH"
python -m src.training.trainer "$@"

# Save results
mkdir -p "$PROJECT_DIR/data/models"
cp -r "$SCRATCH_DIR/data/models/"* "$PROJECT_DIR/data/models/"

echo "Done. Results copied to $PROJECT_DIR/data/models/"
