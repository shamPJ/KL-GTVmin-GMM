#!/bin/bash
#SBATCH --job-name=hetfl_gmm
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --array=0-29
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
VAR_LIST=(0.1 1 10) # variance scale
SEED_LIST=(0 1 2 3 4 5 6 7 8 9)

LR=0.0005
LMD=1
ROUNDS=50
NCLIENTS=10
N_VAL=1000
K=5
OUTDIR=outputs/var

# ===============================
# Index mapping
# ===============================
IDX=$SLURM_ARRAY_TASK_ID

NSC=${#VAR_LIST[@]}
NS=${#SEED_LIST[@]}

VAR_IDX=$(( IDX / NS ))
SEED_IDX=$(( IDX % NS ))

VAR=${VAR_LIST[$VAR_IDX]}
SEED=${SEED_LIST[$SEED_IDX]}

D_LIST=(2 5 10)
N_LIST=(10 25 50 100)

for D in "${D_LIST[@]}"; do
  for N in "${N_LIST[@]}"; do

    echo "========================================"
    echo "D=$D N=$N var_scale=$VAR seed=$SEED"
    echo "========================================"

    srun python scripts/run_means_shift.py \
      --D $D \
      --N $N \
      --N_val $N_VAL \
      --K $K \
      --n_clients $NCLIENTS \
      --p 0.8 \
      --var_scale $VAR \
      --shift_scale 1.0 \
      --cov diag \
      --algo distrGTVMinKL \
      --reg_term $LMD \
      --lrate $LR \
      --rounds $ROUNDS \
      --local_steps 10 \
      --batch_size 256 \
      --m_self 512 \
      --m_nbr 512 \
      --use_self_term \
      --seed $SEED \
      --device cuda \
      --outdir $OUTDIR

  done
done
