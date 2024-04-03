#!/bin/bash
#SBATCH -t 2:00:00
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=31500M        # Memory proportional to GPUs: 31500 Cedar, 63500 Graham.
#SBATCH -J _tcell_fib_try_fastgtn_openhgnn
#SBATCH -N 1
# ---------------------------------------------------------------------
echo "Current working directory: $(pwd)"
echo "Starting run at: $(date)"
# ---------------------------------------------------------------------
echo ""
echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
echo "This is job $SLURM_ARRAY_TASK_ID out of $SLURM_ARRAY_TASK_COUNT jobs."
echo ""
# ---------------------------------------------------------------------
# activate your virtual environment
module load python/3.10
module load scipy-stack
source /project/def-gregorys/almas/spgraph_env/bin/activate
nvidia-smi


cd /home/almas/projects/def-gregorys/almas/OpenHGNN
CUDA_LAUNCH_BLOCKING=1 python main.py -m fastGTN -t node_classification -d tcell_fib -g 0 --use_best_config --early_stopping False # will save model if early stopping is true, need to figure out what to do if early stopping is false
tensorboard --logdir=./openhgnn/output/fastGTN/
# print out all variables and get their dimensions
