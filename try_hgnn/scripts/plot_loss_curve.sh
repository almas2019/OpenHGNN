#!/bin/bash
#SBATCH -t 1:00:00
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=31500M        # Memory proportional to GPUs: 31500 Cedar, 63500 Graham.
#SBATCH -J plot_loss
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


cd /home/almas/projects/def-gregorys/almas/OpenHGNN/try_hgnn/scripts/
python plot_log_vals.py "/project/def-gregorys/almas/OpenHGNN/try_hgnn/data/2024-Mar-28_1034_fastGTN_node_classification_tcell_fib.csv" "Epoch" "Train Loss" "/project/def-gregorys/almas/OpenHGNN/try_hgnn/img" "train_loss_tcell_fib"


