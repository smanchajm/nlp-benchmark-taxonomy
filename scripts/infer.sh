#!/bin/bash -l
#SBATCH --job-name=infer
#SBATCH --partition=rali
#SBATCH --gres=gpu:rtx_a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=30
#SBATCH --output=logs/slurm/%j.out

INPUT="${1:?Usage: sbatch scripts/infer.sh <input_parquet> <checkpoint_path> <config>}"
CHECKPOINT="${2:?Usage: sbatch scripts/infer.sh <input_parquet> <checkpoint_path> <config>}"
CONFIG="${3:?Usage: sbatch scripts/infer.sh <input_parquet> <checkpoint_path> <config>}"

PROJECT_DIR="$HOME/nlp-benchmark-taxonomy"
SCRATCH_DIR="/Tmp/$(whoami)/${SLURM_JOB_ID}"

cleanup() { rm -rf "$SCRATCH_DIR"; echo "Cleaned up $SCRATCH_DIR"; }
trap cleanup EXIT

# Copy project + checkpoint + input data
mkdir -p "$SCRATCH_DIR/$(dirname "$INPUT")"
cp -r "$PROJECT_DIR"/{src,config,requirements-train.txt,pyproject.toml,uv.lock} "$SCRATCH_DIR/"
cp "$PROJECT_DIR/$INPUT" "$SCRATCH_DIR/$INPUT"
mkdir -p "$SCRATCH_DIR/$CHECKPOINT"
if [ -f "$SCRATCH_DIR/$CHECKPOINT/model.safetensors" ]; then
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
python -m src.training.infer "$INPUT" --checkpoint "$CHECKPOINT" --config "$CONFIG"

# Copy predictions back
OUTPUT_NAME="$(basename "${INPUT%.parquet}")_predictions.parquet"
cp "$SCRATCH_DIR/$(dirname "$INPUT")/$OUTPUT_NAME" "$PROJECT_DIR/$(dirname "$INPUT")/" 2>/dev/null || true

echo "Done."
