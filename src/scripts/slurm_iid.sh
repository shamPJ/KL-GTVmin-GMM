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

export PYTHONPATH=$PYTHONPATH:$PWD

# ===============================
# Sweep grids
# ===============================
D_LIST=(2 5 10)
N_LIST=(10 25 50 100)
LAM_LIST=(0 0.5 1.0)
SEED_LIST=(0 1 2 3 4 5 6 7 8 9)

ALGOS=("local" "central" "fl")

LR=0.0005
ROUNDS=100
NCLIENTS=10
N_VAL=1000
K=6

# ===============================
# Index mapping
# ===============================
IDX=$SLURM_ARRAY_TASK_ID

ND=${#D_LIST[@]}
NN=${#N_LIST[@]}
NL=${#LAM_LIST[@]}
NS=${#SEED_LIST[@]}

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
echo "D=$D N=$N LAM=$LAM SEED=$SEED"
echo "========================================"

OUT_DIR="results_iid"
mkdir -p $OUT_DIR

# ===============================
# Algorithm loop
# ===============================
for ALG in "${ALGOS[@]}"; do

    case "$ALG" in
        local)
            ARGS="--algo local"
            ;;
        central)
            ARGS="--algo central"
            ;;
        fl)
            ARGS="--algo fl --lrate $LR --rounds $ROUNDS"
            ;;
    esac

    ALG_DIR="${OUT_DIR}/${ALG}"
    mkdir -p "$ALG_DIR"

    echo "Running ALG=$ALG"

    srun python run.py \
        --algo $ALG \
        --D $D \
        --N $N \
        --N_val $N_VAL \
        --K $K \
        --n_clients $NCLIENTS \
        --reg_term $LAM \
        --seed $SEED \
        --device cuda \
        --outdir "$ALG_DIR" \
        $ARGS

done