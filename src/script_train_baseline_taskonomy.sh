#!/bin/bash  
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=16G  # Requested Memory
#SBATCH -p gpu-preempt	  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 06:00:00  # Job time limit
#SBATCH -o 03_slurm_taskonomy/baseline-taskonomy-%j-%a.out  # %j = job ID
#SBATCH --mail-type=ALL
#SBATCH --job-name=taskonomy-baseline-%j-%a
#SBATCH --array=0-9%10   # 10 jobs at a time

variations=(1 2 3 4 5 6 7 8 9 10)

source ~/miniconda3/etc/profile.d/conda.sh conda activate torch2
casenumidx=$((SLURM_ARRAY_TASK_ID))
echo "Running taskonomy baseline training case_num ${variations[$casenumidx]}"


python train_taskonomy.py --model_type "linear" --case_num ${variations[$casenumidx]} --link_pred_src "link_pred_new80" --ckpts_dir "../ckpts_taskonomy_80"
python train_taskonomy.py --model_type "linear" --case_num ${variations[$casenumidx]} --link_pred_src "link_pred_new50" --ckpts_dir "../ckpts_taskonomy_50"

python train_taskonomy.py --model_type "mlp" --case_num ${variations[$casenumidx]} --link_pred_src "link_pred_new80" --ckpts_dir "../ckpts_taskonomy_80"
python train_taskonomy.py --model_type "mlp" --case_num ${variations[$casenumidx]} --link_pred_src "link_pred_new50" --ckpts_dir "../ckpts_taskonomy_50"
