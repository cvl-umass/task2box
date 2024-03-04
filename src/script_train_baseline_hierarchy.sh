#!/bin/bash  
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=12G  # Requested Memory
#SBATCH -p cpu	  # Partition
#SBATCH -G 0  # Number of GPUs
#SBATCH -t 24:00:00  # Job time limit
#SBATCH -o 03_slurm_baseline/baseline-%j-%a.out  # %j = job ID
#SBATCH --mail-type=ALL
#SBATCH --job-name=baseline-%j-%a
#SBATCH --array=0-9%10   # 10 jobs at a time

variations=(1 2 3 4 5 6 7 8 9 10)

source ~/miniconda3/etc/profile.d/conda.sh conda activate torch2
casenumidx=$((SLURM_ARRAY_TASK_ID))
echo "Running baseline training case_num ${variations[$casenumidx]}"

# For Baseline MLP (ImageNet)
python train_baseline_hierarchy.py --model_type mlp --seed 123 --dataset imagenet --gt_pairs_fp ../data/hierarchy_imagenet.csv --ckpts_dir ../ckpts_imagenet_80 --feat_type "clip_gauss" --case_num ${variations[$casenumidx]} --link_pred_src "link_pred_new80"
python train_baseline_hierarchy.py --model_type mlp --seed 123 --dataset imagenet --gt_pairs_fp ../data/hierarchy_imagenet.csv --ckpts_dir ../ckpts_imagenet_80 --feat_type "clip_gauss" --case_num ${variations[$casenumidx]} --link_pred_src "link_pred_new50"
python train_baseline_hierarchy.py --model_type mlp --seed 123 --dataset imagenet --gt_pairs_fp ../data/hierarchy_imagenet.csv --ckpts_dir ../ckpts_imagenet_80 --feat_type "clip_ave" --case_num ${variations[$casenumidx]} --link_pred_src "link_pred_new80"
python train_baseline_hierarchy.py --model_type mlp --seed 123 --dataset imagenet --gt_pairs_fp ../data/hierarchy_imagenet.csv --ckpts_dir ../ckpts_imagenet_80 --feat_type "clip_ave" --case_num ${variations[$casenumidx]} --link_pred_src "link_pred_new50"
python train_baseline_hierarchy.py --model_type mlp --seed 123 --dataset imagenet --gt_pairs_fp ../data/hierarchy_imagenet.csv --ckpts_dir ../ckpts_imagenet_80 --feat_type "fim" --case_num ${variations[$casenumidx]} --link_pred_src "link_pred_new80"
python train_baseline_hierarchy.py --model_type mlp --seed 123 --dataset imagenet --gt_pairs_fp ../data/hierarchy_imagenet.csv --ckpts_dir ../ckpts_imagenet_80 --feat_type "fim" --case_num ${variations[$casenumidx]} --link_pred_src "link_pred_new50"

# For Baseline MLP (CUB+iNat)
python train_baseline_hierarchy.py --model_type mlp --seed 123 --dataset cubinat --gt_pairs_fp ../data/hierarchy_cubinat.csv --ckpts_dir ../ckpts_cubinat_80 --feat_type "clip_gauss" --case_num ${variations[$casenumidx]} --link_pred_src "link_pred_new80"
python train_baseline_hierarchy.py --model_type mlp --seed 123 --dataset cubinat --gt_pairs_fp ../data/hierarchy_cubinat.csv --ckpts_dir ../ckpts_cubinat_50 --feat_type "clip_gauss" --case_num ${variations[$casenumidx]} --link_pred_src "link_pred_new50"
python train_baseline_hierarchy.py --model_type mlp --seed 123 --dataset cubinat --gt_pairs_fp ../data/hierarchy_cubinat.csv --ckpts_dir ../ckpts_cubinat_80 --feat_type "clip_ave" --case_num ${variations[$casenumidx]} --link_pred_src "link_pred_new80"
python train_baseline_hierarchy.py --model_type mlp --seed 123 --dataset cubinat --gt_pairs_fp ../data/hierarchy_cubinat.csv --ckpts_dir ../ckpts_cubinat_50 --feat_type "clip_ave" --case_num ${variations[$casenumidx]} --link_pred_src "link_pred_new50"
python train_baseline_hierarchy.py --model_type mlp --seed 123 --dataset cubinat --gt_pairs_fp ../data/hierarchy_cubinat.csv --ckpts_dir ../ckpts_cubinat_80 --feat_type "fim" --case_num ${variations[$casenumidx]} --link_pred_src "link_pred_new80"
python train_baseline_hierarchy.py --model_type mlp --seed 123 --dataset cubinat --gt_pairs_fp ../data/hierarchy_cubinat.csv --ckpts_dir ../ckpts_cubinat_50 --feat_type "fim" --case_num ${variations[$casenumidx]} --link_pred_src "link_pred_new50"


# For Baseline Linear (ImageNet)
python train_baseline_hierarchy.py --model_type linear --seed 123 --dataset imagenet --gt_pairs_fp ../data/hierarchy_imagenet.csv --ckpts_dir ../ckpts_imagenet_80 --feat_type "clip_gauss" --case_num ${variations[$casenumidx]} --link_pred_src "link_pred_new80"
python train_baseline_hierarchy.py --model_type linear --seed 123 --dataset imagenet --gt_pairs_fp ../data/hierarchy_imagenet.csv --ckpts_dir ../ckpts_imagenet_80 --feat_type "clip_gauss" --case_num ${variations[$casenumidx]} --link_pred_src "link_pred_new50"
python train_baseline_hierarchy.py --model_type linear --seed 123 --dataset imagenet --gt_pairs_fp ../data/hierarchy_imagenet.csv --ckpts_dir ../ckpts_imagenet_80 --feat_type "clip_ave" --case_num ${variations[$casenumidx]} --link_pred_src "link_pred_new80"
python train_baseline_hierarchy.py --model_type linear --seed 123 --dataset imagenet --gt_pairs_fp ../data/hierarchy_imagenet.csv --ckpts_dir ../ckpts_imagenet_80 --feat_type "clip_ave" --case_num ${variations[$casenumidx]} --link_pred_src "link_pred_new50"
python train_baseline_hierarchy.py --model_type linear --seed 123 --dataset imagenet --gt_pairs_fp ../data/hierarchy_imagenet.csv --ckpts_dir ../ckpts_imagenet_80 --feat_type "fim" --case_num ${variations[$casenumidx]} --link_pred_src "link_pred_new80"
python train_baseline_hierarchy.py --model_type linear --seed 123 --dataset imagenet --gt_pairs_fp ../data/hierarchy_imagenet.csv --ckpts_dir ../ckpts_imagenet_80 --feat_type "fim" --case_num ${variations[$casenumidx]} --link_pred_src "link_pred_new50"

# For Baseline Linear (CUB+iNat)
python train_baseline_hierarchy.py --model_type linear --seed 123 --dataset cubinat --gt_pairs_fp ../data/hierarchy_cubinat.csv --ckpts_dir ../ckpts_cubinat_80 --feat_type "clip_gauss" --case_num ${variations[$casenumidx]} --link_pred_src "link_pred_new80"
python train_baseline_hierarchy.py --model_type linear --seed 123 --dataset cubinat --gt_pairs_fp ../data/hierarchy_cubinat.csv --ckpts_dir ../ckpts_cubinat_50 --feat_type "clip_gauss" --case_num ${variations[$casenumidx]} --link_pred_src "link_pred_new50"
python train_baseline_hierarchy.py --model_type linear --seed 123 --dataset cubinat --gt_pairs_fp ../data/hierarchy_cubinat.csv --ckpts_dir ../ckpts_cubinat_80 --feat_type "clip_ave" --case_num ${variations[$casenumidx]} --link_pred_src "link_pred_new80"
python train_baseline_hierarchy.py --model_type linear --seed 123 --dataset cubinat --gt_pairs_fp ../data/hierarchy_cubinat.csv --ckpts_dir ../ckpts_cubinat_50 --feat_type "clip_ave" --case_num ${variations[$casenumidx]} --link_pred_src "link_pred_new50"
python train_baseline_hierarchy.py --model_type linear --seed 123 --dataset cubinat --gt_pairs_fp ../data/hierarchy_cubinat.csv --ckpts_dir ../ckpts_cubinat_80 --feat_type "fim" --case_num ${variations[$casenumidx]} --link_pred_src "link_pred_new80"
python train_baseline_hierarchy.py --model_type linear --seed 123 --dataset cubinat --gt_pairs_fp ../data/hierarchy_cubinat.csv --ckpts_dir ../ckpts_cubinat_50 --feat_type "fim" --case_num ${variations[$casenumidx]} --link_pred_src "link_pred_new50"