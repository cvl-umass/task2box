from box_embeddings.parameterizations.box_tensor import *
from box_embeddings.modules.volume.soft_volume import soft_volume
from box_embeddings.modules.intersection import hard_intersection, gumbel_intersection
# Initialize and Training
import numpy as np
from box_embeddings.parameterizations.box_tensor import BoxTensor
from box_embeddings.modules.volume.volume import Volume
from box_embeddings.modules.intersection import Intersection

import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
from datetime import datetime
import os
import pickle
import pandas as pd
from tqdm import tqdm
from loguru import logger
import glob
import argparse

from utils.model import BoxModel

DATE_STR = datetime.now().strftime("%Y%m%d-%H%M%S")

parser = argparse.ArgumentParser(description='Evaluation Script for Hierarchy on Existing Datasets')
parser.add_argument('--dataset', default="cubinat", type=str,
                    help='Dataset to use (cubinat, imagenet)')
parser.add_argument('--link_pred_src', default="link_pred_new80", type=str,
                    help='Where to get train data')
parser.add_argument('--gt_pairs_fp', default="../data/hierarchy_both.csv", type=str,
                    help='Directory for saving results')
parser.add_argument('--ckpts_dir', default="../ckpts_cubinat_80", type=str,
                    help='Directory where dataset cache is downloaded/stored')
parser.add_argument('--results_dir', default="../results", type=str,
                    help='Directory where results are saved')
parser.add_argument('--batch_size', default=16, type=int,
                    help='batch size for linear/mlp evals. ')

parser.add_argument('--epoch', default=1350, type=int,
                    help='Epoch to use for evaluation')
parser.add_argument('--case_num', default=1, type=int,
                    help='Case number for training/evaluation [1-10] ')
parser.add_argument('--feat_type', default="clip_gauss", type=str,
                    help='Model used for extracting embeddings ("clip_ave", "clip_gauss", "fim")')
parser.add_argument('--model_type', default="Task2Box", type=str,
                    help='Name of model to evaluate ["Task2Box", "linear", "mlp", "random", "weighted_random"]')
parser.add_argument('--box_dim', default=2, type=int,
                    help='Box dim to use for evaluation')


def get_metric_for_random_links(datasets, mask_idxs, filtered_gt_pairs, p=[0.5, 0.5]):
    y_pred, y_true = [], []
    for idx1, idx2 in mask_idxs:
        d1, d2 = datasets[idx1], datasets[idx2]
        label = 0
        relevant_rows = filtered_gt_pairs[((filtered_gt_pairs["parent"]==d1)&(filtered_gt_pairs["child"]==d2))]
        num_rows = len(relevant_rows)
        if num_rows > 0:
            label = 1
        pred = np.random.choice([0, 1], p=p)
        y_pred.append(pred)
        y_true.append(label)
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    TP = np.sum((y_pred==1)&(y_true==1))
    FP = np.sum((y_pred==1)&(y_true==0))
    FN = np.sum((y_pred==0)&(y_true==1))
    recall = TP/(TP+FN) if (TP+FN) else 0
    precision = TP/(TP+FP) if (TP+FP) else 0
    f1 = (2*precision*recall)/(precision+recall) if (precision+recall) else 0

    return f1, precision, recall


def get_metrics(mask_idxs, box_per_dataset, datasets, filtered_gt_pairs, box_vol, box_int, thresh):
    y_pred, y_true = [], []
    for idx1, idx2 in mask_idxs:
        d1, d2 = datasets[idx1], datasets[idx2]
        label = 0
        relevant_rows = filtered_gt_pairs[((filtered_gt_pairs["parent"]==d1)&(filtered_gt_pairs["child"]==d2))]
        num_rows = len(relevant_rows)
        if num_rows > 0:
            label = 1
        box_p, box_c = box_per_dataset[d1], box_per_dataset[d2]

        overlap_pc = torch.exp(box_vol(box_int(box_p, box_c)) - box_vol(box_p))
        overlap_cp = torch.exp(box_vol(box_int(box_c, box_p)) - box_vol(box_c))
        pred = 0
        if (overlap_cp > overlap_pc) and (overlap_cp > thresh):
            pred = 1
        y_pred.append(pred)
        y_true.append(label)
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    TP = np.sum((y_pred==1)&(y_true==1))
    FP = np.sum((y_pred==1)&(y_true==0))
    FN = np.sum((y_pred==0)&(y_true==1))
    recall = TP/(TP+FN) if (TP+FN) else 0
    precision = TP/(TP+FP) if (TP+FP) else 0
    f1 = (2*precision*recall)/(precision+recall) if (precision+recall) else 0

    return f1, precision, recall

def evaluate(model, x, datasets, mask_idxs, filtered_gt_pairs, box_vol, box_int, thresh=None):
     with torch.no_grad():
        ll_coords, box_sizes = model(x)
        ur_coords = ll_coords + box_sizes
        boxes = torch.stack((ll_coords, ur_coords), dim=1)

        box_per_dataset = {}
        for d_idx, d in enumerate(datasets):
            box_per_dataset[d] = BoxTensor(boxes[d_idx, :, :])

        if thresh is None:  # means need to find threshold (should use validation set)
            threshes = [x/10. for x in range(1,10)]
            logger.debug(f"Threshold not available. Finding optimal among {threshes}")
            val_f1s, val_precs, val_recs = [], [], []
            for t in threshes:
                val_f1, val_prec, val_rec = get_metrics(
                    mask_idxs, box_per_dataset, datasets, filtered_gt_pairs, box_vol, box_int, t
                )
                val_f1s.append(val_f1)
                val_precs.append(val_prec)
                val_recs.append(val_rec)
            try:
                optimal_idx = np.nanargmax(val_f1s)
            except ValueError:
                optimal_idx = 0
            optimal_threshold = threshes[optimal_idx]
            logger.debug(f"Found optimal threshold: {optimal_threshold}. f1 score: {val_f1s[optimal_idx]}. Candidates: {val_f1s}")
            return val_f1s, val_precs, val_recs, optimal_threshold
        else:
            test_f1, test_prec, test_rec = get_metrics(
                mask_idxs, box_per_dataset, datasets, filtered_gt_pairs, box_vol, box_int, thresh
            )
            logger.debug(f"Evaluated with threshold={thresh}. f1: {test_f1}, prec: {test_prec}, rec: {test_rec}")
            return test_f1, test_prec, test_rec

def evaluate_linear(args, model, val_loader, thresh=None):
    model.eval()
    with torch.no_grad():
        all_preds = []
        all_labels = []
        for x,y in val_loader:
            y = y.type(torch.FloatTensor)
            preds = model(x)
            pred_np = preds.detach().cpu().numpy()
            y_np = torch.squeeze(y).detach().cpu().numpy()
            all_preds.append(np.reshape(pred_np, (-1,)))
            # print(f"preds: {preds.shape} pred_np: {pred_np.shape}")
            all_labels.append(np.reshape(y_np, (-1,)))
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        logger.debug(f"all_preds: {all_preds.shape} all_labels: {all_labels.shape}")
        
        if thresh is None:  # means need to find threshold (should use validation set)
            threshes = [x/10. for x in range(1,10)]
            logger.debug(f"Threshold not available. Finding optimal among {threshes}")
            val_f1s, val_precs, val_recs = [], [], []
            for t in threshes:
                val_f1, val_prec, val_rec = get_linear_metrics(all_preds, all_labels, thresh=t)
                val_f1s.append(val_f1)
                val_precs.append(val_prec)
                val_recs.append(val_rec)
            optimal_idx = np.nanargmax(val_f1s)
            optimal_threshold = threshes[optimal_idx]
            logger.debug(f"Found optimal threshold: {optimal_threshold}. f1 score: {val_f1s[optimal_idx]}. Candidates: {val_f1s}")
            return optimal_threshold, val_f1s, val_precs, val_recs
        else:
            test_f1, test_prec, test_rec = get_linear_metrics(all_preds, all_labels, thresh=thresh)
            logger.debug(f"Evaluated with threshold={thresh}. f1: {test_f1}, prec: {test_prec}, rec: {test_rec}")
            return test_f1, test_prec, test_rec

def get_linear_metrics(y_pred, y_true, thresh=0.5):
    y_pred = np.where(y_pred>thresh, 1, 0)
    y_true = np.array(y_true)
    TP = np.sum((y_pred==1)&(y_true==1))
    FP = np.sum((y_pred==1)&(y_true==0))
    FN = np.sum((y_pred==0)&(y_true==1))
    recall = TP/(TP+FN) if (TP+FN) else 0
    precision = TP/(TP+FP) if (TP+FP) else 0
    f1 = (2*precision*recall)/(precision+recall) if (precision+recall) else 0

    return f1, precision, recall

def get_label_from_pair_idxs(mask_idxs, gt_data_pairs, datasets_train):
    labels = []
    for idx1, idx2 in mask_idxs:
        d1, d2 = datasets_train[idx1], datasets_train[idx2]
        relevant_rows = gt_data_pairs[((gt_data_pairs["parent"]==d1)&(gt_data_pairs["child"]==d2))]
        num_rows = len(relevant_rows)
        label = 0
        if num_rows>0:
            label = 1
        labels.append(label)
    assert len(labels) == len(mask_idxs)
    return labels

def get_prob_1s(mask_idxs, datasets, gt_data_pairs):
    num_neg = 0
    num_pos = 0
    for idx1, idx2 in mask_idxs:
        d1, d2 = datasets[idx1], datasets[idx2]
        label = 0
        relevant_rows = gt_data_pairs[((gt_data_pairs["parent"]==d1)&(gt_data_pairs["child"]==d2))]
        num_rows = len(relevant_rows)
        if num_rows > 0:
            label = 1

        if label == 0:
            num_neg += 1
        else:
            num_pos += 1
    return num_pos/(num_pos + num_neg)

if __name__=="__main__":
    args = parser.parse_args()
    logger.debug(f"args: {args}")

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir, exist_ok=True)
    np.random.seed(123)
    random.seed(123)
    torch.manual_seed(123)

    gt_data_pairs = pd.read_csv(args.gt_pairs_fp, index_col=0)
    gt_data_pairs.dropna(inplace=True)

    box_vol = Volume(volume_temperature=0.1, intersection_temperature=0.0001)
    box_int = Intersection(intersection_temperature=0.0001)
    

    all_metrics = []

    # Get masks for train/val/test split for the given case
    if args.dataset == "cubinat":
        data_fp = f"../embeddings/{args.link_pred_src}/case{args.case_num:02d}.pickle"
    elif args.dataset == "imagenet":
        data_fp = f"../embeddings/{args.link_pred_src}/imagenet_case{args.case_num:02d}.pickle"
    else:
        raise NotImplementedError
    with open(data_fp, "rb") as f:
        all_data = pickle.load(f)
    mask = all_data["mask"]
    val_mask = (mask==1).astype(int)
    test_mask = (mask==2).astype(int)
    if args.dataset == "cubinat":   # Load datasets
        datasets = all_data["datasets"]
    elif args.dataset == "imagenet":
        datasets = all_data["train_datasets"]

    tmp_val_mask = np.where(val_mask>0)
    val_mask_idxs = list(zip(tmp_val_mask[0], tmp_val_mask[1]))
    tmp_test_mask = np.where(test_mask>0)
    test_mask_idxs = list(zip(tmp_test_mask[0], tmp_test_mask[1]))
    if args.feat_type == "random":
        test_f1, test_prec, test_rec = get_metric_for_random_links(datasets, test_mask_idxs, gt_data_pairs, p=[0.5, 0.5])
        logger.debug(f"random case {args.case_num:02d} test_f1: {test_f1}, test_prec: {test_prec}, test_rec: {test_rec}")
        all_metrics.append(["random", None, None, args.case_num, test_f1, test_prec, test_rec, None, args.link_pred_src.split("new")[-1], None])
        df = pd.DataFrame(all_metrics, columns=["model", "feat_type", "box_dim", "case_num", "f1", "prec", "rec", "thresh", "supervision_perc", "epoch"])
        df.to_csv(f"{args.results_dir}/{args.model_type}_{args.feat_type}_{args.box_dim}_{args.case_num}_hierarchy_{args.link_pred_src}.csv")
        exit()
    elif args.feat_type == "weighted_random":
        prob_one = get_prob_1s(test_mask_idxs, datasets, gt_data_pairs)
        prob_zero = 1-prob_one
        test_f1, test_prec, test_rec = get_metric_for_random_links(datasets, test_mask_idxs, gt_data_pairs, p=[prob_zero, prob_one])
        logger.debug(f"weighted_random case {args.case_num:02d} test_f1: {test_f1}, test_prec: {test_prec}, test_rec: {test_rec}")
        all_metrics.append(["weighted_random", None, None, args.case_num, test_f1, test_prec, test_rec, None, args.link_pred_src.split("new")[-1], None])
        df = pd.DataFrame(all_metrics, columns=["model", "feat_type", "box_dim", "case_num", "f1", "prec", "rec", "thresh", "supervision_perc", "epoch"])
        df.to_csv(f"{args.results_dir}/{args.model_type}_{args.feat_type}_{args.box_dim}_{args.case_num}_hierarchy_{args.link_pred_src}.csv")
        exit()

    # Load input
    in_feats = all_data[f"{args.feat_type}_train_data"]
    x_input = torch.from_numpy(in_feats).type(torch.FloatTensor)

    # Load checkpoint
    epoch = args.epoch
    if args.model_type == "Task2Box":
        all_ckpts = glob.glob(os.path.join(
            args.ckpts_dir, f'*_{args.feat_type}_case{args.case_num:02d}_{args.epoch:02d}_box{args.box_dim:02d}_model.pth'
        ))
        logger.debug(f"Looking for checkpoints with pattern: *_{args.feat_type}_case{args.case_num:02d}_{args.epoch:02d}_box{args.box_dim:02d}_model.pth")
    elif args.model_type in ["mlp", "linear"]:
        all_ckpts = glob.glob(os.path.join(
            args.ckpts_dir, f'*_{args.model_type}_{args.feat_type}_case{args.case_num:02d}_{rgs.epoch:02d}_model.pth'
        ))
        logger.debug(f"Looking for checkpoints with pattern: *_{args.model_type}_{args.feat_type}_case{args.case_num:02d}_{rgs.epoch:02d}_model.pth")
    all_ckpts = sorted(all_ckpts)
    
    logger.debug(f"all_ckpts: {all_ckpts}")
    if not (len(all_ckpts) >= 1):
        for e in list(range(1500,0,-75)):
            if args.model_type == "Task2Box":
                new_pattern = f'*_{args.feat_type}_case{args.case_num:02d}_{e:02d}_box{args.box_dim:02d}_model.pth'
            elif args.model_type in ["mlp", "linear"]:
                new_pattern = f'*_{args.model_type}_{args.feat_type}_case{args.case_num:02d}_{e:02d}_model.pth'
            logger.error(f"Checkpoint not found. Using new pattern: {new_pattern}")
            all_ckpts = glob.glob(os.path.join(args.ckpts_dir, new_pattern))
            all_ckpts = sorted(all_ckpts)
            if len(all_ckpts) >= 1:
                epoch = e
                logger.warning(f"Using epoch={epoch} instead. Choices: {all_ckpts}")
                break
        else:
            logger.error("No alternative checkpoints found. Exiting.")
            exit()
    # Do for all checkpoints
    for ckpt_fp in all_ckpts:
        logger.debug(f"Using checkpoint {ckpt_fp}")
        ckpt = torch.load(ckpt_fp)

        if args.feat_type == "fim":
            in_dim = 17024
        elif args.feat_type == "clip_gauss":
            in_dim = 4096
        elif args.feat_type == "clip_ave":
            in_dim = 2048
        
        if args.model_type == "Task2Box":
            model = BoxModel(in_dim=in_dim, box_dim=args.box_dim)
        elif args.model_type == "linear":
            model = LinearModel(in_dim=in_dim*2)
        elif args.model_type == "mlp":
            model = MLPModel2(in_dim=in_dim*2)
        else:
            raise NotImplementedError
        model.load_state_dict(ckpt["model_state_dict"], strict=True)

        if args.model_type in ["linear", "mlp"]:
            logger.debug(f"Getting labels from pair idxs")
            val_labels = get_label_from_pair_idxs(val_mask_idxs, gt_data_pairs, datasets)
            test_labels = get_label_from_pair_idxs(test_mask_idxs, gt_data_pairs, datasets)
    
            val_dataset = PairData(in_feats, val_mask_idxs, val_labels)
            val_loader = torch.utils.data.DataLoader(
                dataset=val_dataset,
                batch_size=args.batch_size,
                shuffle=False, num_workers=0)
            test_dataset = PairData(in_feats, test_mask_idxs, test_labels)
            test_loader = torch.utils.data.DataLoader(
                dataset=test_dataset,
                batch_size=args.batch_size,
                shuffle=False, num_workers=0)
            optim_thresh, val_f1s, val_precs, val_recs = evaluate_linear(None, model, val_loader, thresh=None)
            logger.debug(f"Optim thresh={optim_thresh}. All f1s: {val_f1s}. val_precs: {val_precs}. val_recs: {val_recs}")
            test_f1, test_prec, test_rec = evaluate_linear(None, model, test_loader, thresh=optim_thresh)

            all_metrics.append([args.model_type, args.feat_type, None, args.case_num, test_f1, test_prec, test_rec, optim_thresh, args.link_pred_src.split("new")[-1], epoch, ckpt_fp])
        else:
            # Run model on input embeddings
            model.eval()
            val_f1s, val_precs, val_recs, optim_thresh = evaluate(
                model, x_input, datasets, val_mask_idxs, gt_data_pairs, box_vol, box_int, thresh=None
            )
            logger.debug(f"Optim thresh={optim_thresh}. All f1s: {val_f1s}. val_precs: {val_precs}. val_recs: {val_recs}")

            test_f1, test_prec, test_rec= evaluate(
                model, x_input, datasets, test_mask_idxs, gt_data_pairs, box_vol, box_int, thresh=optim_thresh
            )
            all_metrics.append([args.model_type, args.feat_type, args.box_dim, args.case_num, test_f1, test_prec, test_rec, optim_thresh, args.link_pred_src.split("new")[-1], epoch, ckpt_fp])

        df = pd.DataFrame(all_metrics, columns=["model", "feat_type", "box_dim", "case_num", "f1", "prec", "rec", "thresh", "supervision_perc", "epoch", "ckpt_fp"])
        if args.model_type == "Task2Box":
            results_fp = f"{args.results_dir}/{args.model_type}_{args.feat_type}_{args.box_dim}_{args.case_num}_hierarchy_{args.link_pred_src}.csv"
        else:
            results_fp = f"{args.results_dir}/{args.model_type}_{args.feat_type}_{args.case_num}_hierarchy_{args.link_pred_src}.csv"
        df.to_csv(results_fp)