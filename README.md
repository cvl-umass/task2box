# Task2Box: Box Embeddings for Modeling Asymmetric Task Relationships


## Requirements
1. Create an environment with python3.8.16 and activate. Further steps require this environment to be activated.
```
conda create -n task2box python=3.8.16
conda activate task2box
```
2. Install cuda 11.7 and pytorch2.0.1:
```
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```
3. Install additional packages with: `pip install -r requirements.txt`

## Downloading Required Data and Embeddings
1. Download data [here](https://drive.google.com/file/d/1SRwfXaqkdeGraKaT_XcXXsqFE83VV2l7/view?usp=sharing) and place in `data/`
2. Download embeddings [here](https://drive.google.com/file/d/1_YRuWlzfWML5gf1wXGiPluezyWdDqzQp/view?usp=sharing) with all embeddings in `embeddings/`


## Train Task2Box
The following instructions are to be run inside `src/`
### Hierarchy
The training for existing datasets can be run with the following. Note that `box_dim` is specified to be 2 here (which means 2-dim boxes will be used) and trained on case 1. Ten cases are made available for training (case_num [1-10]). `feat_type` can be one of [`clip_gauss`, `clip_ave`, `fim`].
```
# For CUB+iNat
python train_hierarchy.py --dataset cubinat --gt_pairs_fp ../data/hierarchy_cubinat.csv --ckpts_dir ../ckpts_cubinat_80 --box_dim 2 --case_num 1 --feat_type "clip_gauss"

# For ImageNet
python train_hierarchy.py --dataset imagenet --gt_pairs_fp ../data/hierarchy_imagenet.csv --ckpts_dir ../ckpts_imagenet_80 --box_dim 2 --case_num 1 --feat_type "clip_gauss"
```
### Taskonomy
Similar to hierarchical datasets, `case_num` and `box_dim` can be changed as needed. `model_type` can be changed into one of [`Task2Box`, `linear`, `mlp`]
```
python train_taskonomy.py --model_type "Task2Box" --case_num 5 --box_dim 2 --ckpts_dir "../ckpts_taskonomy_80"
```

## Evaluate Task2Box
The following instructions are to be run inside `src/`
### Hierarchy
For existing datasets, the following can be run:
```
# For CUB+iNat
python eval_hierarchy.py  --box_dim 2 --case_num 1 --feat_type "clip_gauss" --model_type "Task2Box" --dataset cubinat --gt_pairs_fp ../data/hierarchy_cubinat.csv --ckpts_dir ../ckpts_cubinat_80 --results_dir "../results_cubinat"

# For ImageNet
python eval_hierarchy.py  --box_dim 2 --case_num 1 --feat_type "clip_gauss" --model_type "Task2Box" --dataset imagenet --gt_pairs_fp ../data/hierarchy_imagenet.csv --ckpts_dir ../ckpts_imagenet_80 --results_dir "../results_imagenet"
```

For novel datasets, the following can be run:
```
# For CUB+iNat
python eval_hierarchy_novel.py  --box_dim 2 --case_num 1 --feat_type "clip_gauss" --model_type "Task2Box" --dataset cubinat --gt_pairs_fp ../data/hierarchy_cubinat.csv --ckpts_dir ../ckpts_cubinat_80 --results_dir "../results_cubinat_novel"

# For ImageNet
python eval_hierarchy_novel.py  --box_dim 2 --case_num 1 --feat_type "clip_gauss" --model_type "Task2Box" --dataset imagenet --gt_pairs_fp ../data/hierarchy_imagenet.csv --ckpts_dir ../ckpts_imagenet_80 --results_dir "../results_imagenet_novel" --fim_dir ../embeddings/task2vec_imagenet/ --gauss_fp ../embeddings/imagenet_clip_gauss.pickle
```

Note that the above scripts produce the metrics for a specific case, feature type, model type, and box dim. To compile all results (for multiple cases) in a directory into a single csv file, the following can be run:
```
python compile_hierarchy_results.py --results_dir "../results_imagenet_novel" --compiled_results_dir "../compiled_results"
```

### Taskonomy
The following will output the compiled results in a single file.
```
python eval_taskonomy.py
```

