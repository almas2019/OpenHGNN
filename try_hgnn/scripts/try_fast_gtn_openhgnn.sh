#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1        # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=31500M        # Memory proportional to GPUs: 31500 Cedar, 63500 Graham.
#SBATCH -J try_fastgtn_openhgnn
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
python main.py -m fastGTN -t node_classification -d imdb4GTN -g 0 --use_best_config
tensorboard --logdir=./openhgnn/output/fastGTN/

