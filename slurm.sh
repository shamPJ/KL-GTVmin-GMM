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
module load cuda    # if required on your cluster
source activate hetFL

# ===============================
# Sweep grids
# ===============================
D_LIST=(2 4 8 16)
N_LIST=(10 25 50)
LAM_LIST=(0 0.5 1.0)
SEED_LIST=(0 1 2 3 4 5 6 7 8 9)

LR=0.003
ROUNDS=100
NCLIENTS=10
K=3

# ===============================
# Index mapping
# ===============================
IDX=$SLURM_ARRAY_TASK_ID

ND=${#D_LIST[@]} # number of elements in the array
NN=${#N_LIST[@]}
NL=${#LAM_LIST[@]}
NS=${#SEED_LIST[@]}

# Total jobs = 4 * 3 * 3 * 10 = 360

D_IDX=$(( IDX / (NN * NL * NS) ))
REM1=$(( IDX % (NN * NL * NS) ))

N_IDX=$(( REM1 / (NL * NS) ))
REM2=$(( REM1 % (NL * NS) ))

LAM_IDX=$(( REM2 / NS ))
SEED_IDX=$(( REM2 % NS ))

D=${D_LIST[$D_IDX]}
N=${N_LIST[$N_IDX]}
LAM=${LAM_LIST[$LAM_IDX]}
SEED=${SEED_LIST[$SEED_IDX]}

echo "========================================"
echo "Running experiment on GPU"
echo "  D       = $D"
echo "  N       = $N"
echo "  lambda  = $LAM"
echo "  seed    = $SEED"
echo "  GPU     = $CUDA_VISIBLE_DEVICES"
echo "========================================"

# ===============================
# Run experiment
# ===============================
srun python run.py \
    --D $D \
    --N $N \
    --K $K \
    --n_clients $NCLIENTS \
    --reg_term $LAM \
    --lrate $LR \
    --rounds $ROUNDS \
    --seed $SEED \
    --device cuda \
    --outdir results
