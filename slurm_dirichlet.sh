#!/bin/bash
#SBATCH --job-name=hetfl_gmm
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --array=0-359
#SBATCH --output=logs/out_%A_%a.out
#SBATCH --error=logs/err_%A_%a.err

# ===============================
# Environment
# ===============================
module load mamba
source activate pytorch-env

# ===============================
# Sweep grids
# ===============================
D_LIST=(2 4 8)
N_LIST=(10 25 50 100)
ALPHA_LIST=(0.2 1.0 10.0)
SEED_LIST=(0 1 2 3 4 5 6 7 8 9)

LR=0.0005
ROUNDS=100
NCLIENTS=10
N_VAL=1000
K=3

# ===============================
# Index mapping
# ===============================
IDX=$SLURM_ARRAY_TASK_ID

ND=${#D_LIST[@]} # number of elements in the array
NN=${#N_LIST[@]}
NA=${#ALPHA_LIST[@]}
NS=${#SEED_LIST[@]}

# Total jobs = 3 * 4 * 3 * 10 = 360

D_IDX=$(( IDX / (NN * NA * NS) ))
REM1=$(( IDX % (NN * NA * NS) ))

N_IDX=$(( REM1 / (NA * NS) ))
REM2=$(( REM1 % (NA * NS) ))

ALPHA_IDX=$(( REM2 / NS ))
SEED_IDX=$(( REM2 % NS ))

D=${D_LIST[$D_IDX]}
N=${N_LIST[$N_IDX]}
ALPHA=${ALPHA_LIST[$ALPHA_IDX]}
SEED=${SEED_LIST[$SEED_IDX]}

echo "========================================"
echo "Running experiment on GPU"
echo "  D       = $D"
echo "  N       = $N"
echo "  alpha  = $ALPHA"
echo "  seed    = $SEED"
echo "  GPU     = $CUDA_VISIBLE_DEVICES"
echo "========================================"

# ===============================
# Run experiment
# ===============================
srun python run_dirichlet.py \
    --D $D \
    --N $N \
    --N_val $N_VAL \
    --K $K \
    --n_clients $NCLIENTS \
    --alpha $ALPHA \
    --lrate $LR \
    --rounds $ROUNDS \
    --seed $SEED \
    --device cuda \
    --outdir results_dirichlet
