#!/bin/bash
#SBATCH --job-name=hetfl_gmm
#SBATCH --time=20:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --array=0-39
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
LAM_LIST=(0 0.1 1.0 10.0)
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

NL=${#LAM_LIST[@]}
NS=${#SEED_LIST[@]}

LAM_IDX=$(( IDX / NS ))
SEED_IDX=$(( IDX % NS ))

LAM=${LAM_LIST[$LAM_IDX]}
SEED=${SEED_LIST[$SEED_IDX]}

for D in "${D_LIST[@]}"; do
  for N in "${N_LIST[@]}"; do

    echo "========================================"
    echo "D=$D N=$N lambda=$LAM seed=$SEED"
    echo "========================================"

    srun python scripts/run_lmbd.py \
      --D $D \
      --N $N \
      --N_val $N_VAL \
      --K $K \
      --n_clients $NCLIENTS \
      --algo distrGTVMinKL \
      --reg_term $LAM \
      --lrate $LR \
      --rounds $ROUNDS \
      --local_steps 200 \
      --batch_size 256 \
      --m_self 512 \
      --m_nbr 512 \
      --use_forward_term \
      --seed $SEED \
      --device cuda

  done
done
