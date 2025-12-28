#!/bin/bash
#SBATCH --job-name=sfda_visda
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --gpus=1
#SBATCH --time=24:00:00

# Get the directory where the slurm script is located (where sbatch was run from)
SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
echo "Script directory: $SCRIPT_DIR"
echo "Current working directory: $(pwd)"
cd "$SCRIPT_DIR"
echo "Changed to: $(pwd)"

# Activate Environment (auto-detect conda or use local venv)
echo "Running on node: $(hostname)"
source /media02/nnthao15/miniconda3/bin/activate
conda activate "$SCRIPT_DIR/venv"


LOCAL_DATA="/media02/nnthao15/Cuong/C-SFDA/data"

echo "Using data from: $LOCAL_DATA"

# 4. RUN TRAINING
python main_csfda.py \
    train_source=false \
    seed=2022 \
    data.dataset="VISDA-C" \
    data.data_root="$LOCAL_DATA" \
    data.source_domains="[train]" \
    data.target_domains="[validation]" \
    data.batch_size=64 \
    data.workers=4 \
    model_src.arch="resnet101" \
    model_tta.src_log_dir="./checkpoint/VISDA-C" \
    learn.epochs=250 \
    optim.lr=2e-4 \
    multiprocessing_distributed=false \
    use_wandb=false \
    resume="./checkpoint/checkpoint_1_latest_train-validation.pth.tar"

# No cleanup needed
echo "Done!"
