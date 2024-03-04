#!/bin/bash  
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=12G  # Requested Memory
#SBATCH -p gpu-preempt	  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 00:30:00  # Job time limit
#SBATCH -o 10_slurm_evals/hier-novel-%j-%a.out  # %j = job ID
#SBATCH --mail-type=ALL
#SBATCH --job-name=eval-novel-%j-%a
#SBATCH --array=0-49%50   # 50 jobs at a time

variations=(1 2 1 3 1 5 1 10 1 20 2 2 2 3 2 5 2 10 2 20 3 2 3 3 3 5 3 10 3 20 4 2 4 3 4 5 4 10 4 20 5 2 5 3 5 5 5 10 5 20 6 2 6 3 6 5 6 10 6 20 7 2 7 3 7 5 7 10 7 20 8 2 8 3 8 5 8 10 8 20 9 2 9 3 9 5 9 10 9 20 10 2 10 3 10 5 10 10 10 20)


source ~/miniconda3/etc/profile.d/conda.sh conda activate torch2

casenumidx=$((SLURM_ARRAY_TASK_ID*2))
boxdimidx=$((SLURM_ARRAY_TASK_ID*2 + 1))
echo "Running box model eval box_dim ${variations[$boxdimidx]} case_num ${variations[$casenumidx]}"

# Below are for models trained on iNat+CUB
python eval_hierarchy_novel.py  --box_dim ${variations[$boxdimidx]} --case_num ${variations[$casenumidx]} --feat_type "clip_gauss" --model_type "Task2Box" --dataset cubinat --gt_pairs_fp ../data/hierarchy_cubinat.csv --ckpts_dir ../ckpts_cubinat_80 --link_pred_src "link_pred_new80" --results_dir "../results_cubinat_novel"
python eval_hierarchy_novel.py  --box_dim ${variations[$boxdimidx]} --case_num ${variations[$casenumidx]} --feat_type "clip_ave" --model_type "Task2Box" --dataset cubinat --gt_pairs_fp ../data/hierarchy_cubinat.csv --ckpts_dir ../ckpts_cubinat_80 --link_pred_src "link_pred_new80" --results_dir "../results_cubinat_novel"
python eval_hierarchy_novel.py  --box_dim ${variations[$boxdimidx]} --case_num ${variations[$casenumidx]} --feat_type "fim" --model_type "Task2Box" --dataset cubinat --gt_pairs_fp ../data/hierarchy_cubinat.csv --ckpts_dir ../ckpts_cubinat_80 --link_pred_src "link_pred_new80" --results_dir "../results_cubinat_novel"
python eval_hierarchy_novel.py  --box_dim ${variations[$boxdimidx]} --case_num ${variations[$casenumidx]} --feat_type "clip_gauss" --model_type "Task2Box" --dataset cubinat --gt_pairs_fp ../data/hierarchy_cubinat.csv --ckpts_dir ../ckpts_cubinat_50 --link_pred_src "link_pred_new50" --results_dir "../results_cubinat_novel"
python eval_hierarchy_novel.py  --box_dim ${variations[$boxdimidx]} --case_num ${variations[$casenumidx]} --feat_type "clip_ave" --model_type "Task2Box" --dataset cubinat --gt_pairs_fp ../data/hierarchy_cubinat.csv --ckpts_dir ../ckpts_cubinat_50 --link_pred_src "link_pred_new50" --results_dir "../results_cubinat_novel"
python eval_hierarchy_novel.py  --box_dim ${variations[$boxdimidx]} --case_num ${variations[$casenumidx]} --feat_type "fim" --model_type "Task2Box" --dataset cubinat --gt_pairs_fp ../data/hierarchy_cubinat.csv --ckpts_dir ../ckpts_cubinat_50 --link_pred_src "link_pred_new50" --results_dir "../results_cubinat_novel"

# Below are for models trained on imagenet
python eval_hierarchy_novel.py  --box_dim ${variations[$boxdimidx]} --case_num ${variations[$casenumidx]} --feat_type "clip_gauss" --model_type "Task2Box" --dataset imagenet --gt_pairs_fp ../data/hierarchy_imagenet.csv --ckpts_dir ../ckpts_imagenet_80 --link_pred_src "link_pred_new80" --results_dir "../results_imagenet_novel" --fim_dir ../embeddings/task2vec_imagenet/ --gauss_fp ../embeddings/imagenet_clip_gauss.pickle
python eval_hierarchy_novel.py  --box_dim ${variations[$boxdimidx]} --case_num ${variations[$casenumidx]} --feat_type "clip_ave" --model_type "Task2Box" --dataset imagenet --gt_pairs_fp ../data/hierarchy_imagenet.csv --ckpts_dir ../ckpts_imagenet_80 --link_pred_src "link_pred_new80" --results_dir "../results_imagenet_novel" --fim_dir ../embeddings/task2vec_imagenet/ --gauss_fp ../embeddings/imagenet_clip_gauss.pickle
python eval_hierarchy_novel.py  --box_dim ${variations[$boxdimidx]} --case_num ${variations[$casenumidx]} --feat_type "fim" --model_type "Task2Box" --dataset imagenet --gt_pairs_fp ../data/hierarchy_imagenet.csv --ckpts_dir ../ckpts_imagenet_80 --link_pred_src "link_pred_new80" --results_dir "../results_imagenet_novel" --fim_dir ../embeddings/task2vec_imagenet/ --gauss_fp ../embeddings/imagenet_clip_gauss.pickle
python eval_hierarchy_novel.py  --box_dim ${variations[$boxdimidx]} --case_num ${variations[$casenumidx]} --feat_type "clip_gauss" --model_type "Task2Box" --dataset imagenet --gt_pairs_fp ../data/hierarchy_imagenet.csv --ckpts_dir ../ckpts_imagenet_50 --link_pred_src "link_pred_new50" --results_dir "../results_imagenet_novel" --fim_dir ../embeddings/task2vec_imagenet/ --gauss_fp ../embeddings/imagenet_clip_gauss.pickle
python eval_hierarchy_novel.py  --box_dim ${variations[$boxdimidx]} --case_num ${variations[$casenumidx]} --feat_type "clip_ave" --model_type "Task2Box" --dataset imagenet --gt_pairs_fp ../data/hierarchy_imagenet.csv --ckpts_dir ../ckpts_imagenet_50 --link_pred_src "link_pred_new50" --results_dir "../results_imagenet_novel" --fim_dir ../embeddings/task2vec_imagenet/ --gauss_fp ../embeddings/imagenet_clip_gauss.pickle
python eval_hierarchy_novel.py  --box_dim ${variations[$boxdimidx]} --case_num ${variations[$casenumidx]} --feat_type "fim" --model_type "Task2Box" --dataset imagenet --gt_pairs_fp ../data/hierarchy_imagenet.csv --ckpts_dir ../ckpts_imagenet_50 --link_pred_src "link_pred_new50" --results_dir "../results_imagenet_novel" --fim_dir ../embeddings/task2vec_imagenet/ --gauss_fp ../embeddings/imagenet_clip_gauss.pickle

