import numpy as np
import random
import torch
from datetime import datetime
import os
import pickle
import pandas as pd
from tqdm import tqdm
from loguru import logger
import glob
import scipy.stats as stats
import argparse


DATE_STR = datetime.now().strftime("%Y%m%d-%H%M%S")

parser = argparse.ArgumentParser(description='Taskonomy evaluation script')
parser.add_argument('--ckpt_dirs', default=["../ckpts_taskonomy_80"], 
                    nargs='+', help='Checkpoint directory(s) to be evaluated')
parser.add_argument('--model_types', default=["Task2Box", "linear", "mlp"],   # NOTE: random baseline always included by default
                    nargs='+', help='Model type(s) to be evaluated')
parser.add_argument('--case_nums', default=[1,2,3,4,5,6,7,8,9,10], 
                    nargs='+', help='Case(s) to be evaluated')
parser.add_argument('--box_dims', default=[2, 3, 5, 10, 20], 
                    nargs='+', help='Box dimension(s) to be evaluated')
parser.add_argument('--results_dir', default="../results_taskonomy", type=str,
                    help='Directory to save results')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. Default is None but can set to number 123 (orig)')
parser.add_argument('--epoch', default=2999, type=int,
                    help='Epoch to use for evaluation')

def get_mape(y_true, y_pred):
    return np.sum(np.abs(np.abs(y_true - y_pred)/y_true))/(y_true.shape[0])
def get_tanh_inv(x, k=30):
    return 0.5*np.log((1+(x/k))/(1-(x/k)))

if __name__=="__main__":
    args = parser.parse_args()
    logger.debug(f"args: {args}")
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir, exist_ok=True)
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    epoch = args.epoch
    embed_all_metrics = []
    link_pred_all_metrics = []
    for model_type in args.model_types:
        for case_num in args.case_nums:
            for ckpt_dir in args.ckpt_dirs:
                loss_type = None
                if "l1" in ckpt_dir:
                    loss_type = "L1"
                supervision_perc = ckpt_dir.strip("/").split("_")[-1]

                for box_dim in args.box_dims:
                    if model_type == "Task2Box":
                        ckpt_pattern =  f'*_case{case_num:02d}_taskonomy_{epoch:02d}_box{box_dim:02d}_model.pth'
                    elif model_type in ["Task2BoxSmall", "Task2BoxLinear"]:
                        ckpt_pattern =  f'*_{model_type}_case{case_num:02d}_taskonomy_{epoch:02d}_box{box_dim:02d}_model.pth'
                    elif model_type in ["linear", "mlp"]:
                        if box_dim != args.box_dims[0]:
                            continue
                        box_dim = None
                        ckpt_pattern =  f'*_{model_type}_case{case_num:02d}_taskonomy_{epoch:02d}_model.pth'
                    all_ckpts = glob.glob(os.path.join(ckpt_dir, ckpt_pattern))
                    all_ckpts = sorted(all_ckpts)
                    if len(all_ckpts) < 1:
                        for e in list(range(3000,0,-150)):

                            if model_type == "Task2Box":
                                new_pattern =  f'*_case{case_num:02d}_taskonomy_{e:02d}_box{box_dim:02d}_model.pth'
                            elif model_type in ["Task2BoxSmall", "Task2BoxLinear"]:
                                new_pattern =  f'*_{model_type}_case{case_num:02d}_taskonomy_{e:02d}_box{box_dim:02d}_model.pth'
                            elif model_type in ["linear", "mlp"]:
                                if box_dim != args.box_dims[0]:
                                    continue
                                box_dim = None
                                new_pattern =  f'*_{model_type}_case{case_num:02d}_taskonomy_{e:02d}_model.pth'
                            logger.error(f"Checkpoint not found. Using new pattern: {new_pattern}")
                            all_ckpts = glob.glob(os.path.join(ckpt_dir, new_pattern))
                            all_ckpts = sorted(all_ckpts)
                            if len(all_ckpts) >= 1:
                                epoch = e
                                logger.warning(f"Using epoch={epoch} instead. Choices: {all_ckpts}")
                                break
                        else:
                            logger.error(f"Pattern not found. Skipping {ckpt_pattern}. ckpt_dir: {ckpt_dir}")
                            continue
                    for ckpt_fp in all_ckpts:

                        results = torch.load(ckpt_fp)
                        if (box_dim == args.box_dims[0]) and (model_type == args.model_types[0]):
                            # Do random prediction
                            labels = results["test_orig_labels"]
                            random_preds = np.random.rand(labels.shape[0])
                            random_preds = get_tanh_inv(random_preds, k=50)
                            rho, p_value = stats.spearmanr(labels, random_preds)
                            mae = np.sum(np.abs(labels-random_preds))/labels.shape[0]
                            mse = np.sum((labels-random_preds)**2)/labels.shape[0]
                            mape = get_mape(labels, random_preds)
                            link_pred_all_metrics.append(["random", None, case_num, None, None, rho, p_value, mae, mse, mape, supervision_perc, epoch, loss_type, ckpt_fp])

                            _, _, _, _, _, test_new_orig_labels = results["novel_test_metrics"]
                            random_preds = np.random.rand(test_new_orig_labels.shape[0])
                            random_preds = get_tanh_inv(random_preds, k=50)
                            rho, p_value = stats.spearmanr(test_new_orig_labels, random_preds)
                            mae = np.sum(np.abs(test_new_orig_labels-random_preds))/test_new_orig_labels.shape[0]
                            mse = np.sum((test_new_orig_labels-random_preds)**2)/test_new_orig_labels.shape[0]
                            mape = get_mape(test_new_orig_labels, random_preds)
                            embed_all_metrics.append(["random", None, case_num, None, None, rho, p_value, mae, mse, mape, supervision_perc, epoch, loss_type, ckpt_fp])


                        test_rho, test_p_value, test_rho_orig, test_p_value_orig = results["test_metrics"]
                        mae = np.sum(np.abs(results["test_orig_labels"]-results["test_orig_preds"]))/results["test_orig_labels"].shape[0]
                        mse = np.sum((results["test_orig_labels"]-results["test_orig_preds"])**2)/results["test_orig_labels"].shape[0]
                        mape = get_mape(results["test_orig_labels"], results["test_orig_preds"])
                        link_pred_all_metrics.append([model_type, box_dim, case_num, test_rho, test_p_value, test_rho_orig, test_p_value_orig, mae, mse, mape, supervision_perc, epoch, loss_type, ckpt_fp])
                        link_pred_df = pd.DataFrame(link_pred_all_metrics, columns=["model", "box_dim", "case_num", "rho(transformed)", "p_val(transformed)", "rho (orig)", "p_val(orig)", "mae", "mse", "mape", "supervision_perc", "epoch", "loss_type", "ckpt_fp"])
                        link_pred_df.to_csv(f"{args.results_dir}/{DATE_STR}_taskonomy_existing_correlation.csv")

                        test_new_rho, test_new_p_value, test_new_rho_orig, test_new_p_value_orig, test_new_orig_preds, test_new_orig_labels = results["novel_test_metrics"]
                        mae = np.sum(np.abs(test_new_orig_labels-test_new_orig_preds))/test_new_orig_labels.shape[0]
                        mse = np.sum((test_new_orig_labels-test_new_orig_preds)**2)/test_new_orig_labels.shape[0]
                        mape = get_mape(test_new_orig_labels, test_new_orig_preds)
                        embed_all_metrics.append([model_type, box_dim, case_num, test_new_rho, test_new_p_value, test_new_rho_orig, test_new_p_value_orig, mae, mse, mape, supervision_perc, epoch, loss_type, ckpt_fp])
                        embed_df = pd.DataFrame(embed_all_metrics, columns=["model", "box_dim", "case_num", "rho(transformed)", "p_val(transformed)", "rho (orig)", "p_val(orig)", "mae", "mse", "mape", "supervision_perc", "epoch", "loss_type", "ckpt_fp"])
                        embed_df.to_csv(f"{args.results_dir}/{DATE_STR}_taskonomy_novel_correlation.csv")