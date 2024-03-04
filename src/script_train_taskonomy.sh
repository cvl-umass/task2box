#!/bin/bash  
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=16G  # Requested Memory
#SBATCH -p gpu-preempt	  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 06:00:00  # Job time limit
#SBATCH -o 03_slurm_taskonomy/boxmodel-taskonomy-%j-%a.out  # %j = job ID
#SBATCH --mail-type=ALL
#SBATCH --job-name=taskonomy-boxmodel-%j-%a
#SBATCH --array=0-49%50   # 50 jobs at a time

variations=(1 2 1 3 1 5 1 10 1 20 2 2 2 3 2 5 2 10 2 20 3 2 3 3 3 5 3 10 3 20 4 2 4 3 4 5 4 10 4 20 5 2 5 3 5 5 5 10 5 20 6 2 6 3 6 5 6 10 6 20 7 2 7 3 7 5 7 10 7 20 8 2 8 3 8 5 8 10 8 20 9 2 9 3 9 5 9 10 9 20 10 2 10 3 10 5 10 10 10 20)

source ~/miniconda3/etc/profile.d/conda.sh conda activate torch2
casenumidx=$((SLURM_ARRAY_TASK_ID*2))
boxdimidx=$((SLURM_ARRAY_TASK_ID*2 + 1))
echo "Running box model training box_dim ${variations[$boxdimidx]} case_num ${variations[$casenumidx]}"

python train_taskonomy.py --model_type "Task2Box" --case_num ${variations[$casenumidx]} --box_dim ${variations[$boxdimidx]} --link_pred_src "link_pred_new80" --ckpts_dir "../ckpts_taskonomy_80"
python train_taskonomy.py --model_type "Task2Box" --case_num ${variations[$casenumidx]} --box_dim ${variations[$boxdimidx]} --link_pred_src "link_pred_new50" --ckpts_dir "../ckpts_taskonomy_50"
