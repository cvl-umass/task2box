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

from utils.model import BoxModel, MLPModel, LinearModel
from utils.dataloader import EmbedPairData

DATE_STR = datetime.now().strftime("%Y%m%d-%H%M%S")
NUM_TEST_DATASETS = 100 # number of datasets to embed (test set)
NUM_VAL_DATASETS = 10 # number of datasets to embed (validation set)

parser = argparse.ArgumentParser(description='Evaluation Script for Hierarchy on Novel Datasets')
parser.add_argument('--dataset', default="cubinat", type=str,
                    help='Dataset to use (cubinat, imagenet)')
parser.add_argument('--link_pred_src', default="link_pred_new80", type=str,
                    help='Where to get train data')
parser.add_argument('--gt_pairs_fp', default="../data/hierarchy_both.csv", type=str,
                    help='Directory for saving results')
parser.add_argument('--txt_labels_dir', default="../data", type=str,
                    help='Directory for saving results')
parser.add_argument('--ckpts_dir', default="../ckpts_cubinat_80", type=str,
                    help='Directory where dataset cache is downloaded/stored')
parser.add_argument('--results_dir', default="../results_novel", type=str,
                    help='Directory where results are saved')
parser.add_argument('--fim_dir', default="../embeddings/task2vec_cubinat/", type=str,
                    help='Directory where FIM embeddings are saved')
parser.add_argument('--gauss_fp', default="../embeddings/cub_inat2018_clip_gauss.pickle", type=str,
                    help='Filepath of clip_gauss feats')

parser.add_argument('--epoch', default=1350, type=int,
                    help='Epoch to use for evaluation')
parser.add_argument('--case_num', default=1, type=int,
                    help='Case number for training/evaluation [1-10] ')
parser.add_argument('--feat_type', default="clip_gauss", type=str,
                    help='Model used for extracting embeddings ("clip_ave", "clip_gauss", "fim")')
parser.add_argument('--model_type', default="Task2Box", type=str,
                    help='Name of model to evaluate. choices: ["Task2Box", "mlp", "linear", "random", "weighted_random"]')
parser.add_argument('--box_dim', default=2, type=int,
                    help='Box dim to use for evaluation')
parser.add_argument('--batch_size', default=16, type=int,
                    help='batch size for linear/mlp evals. ')

def get_embed_metrics(train_datasets, test_datasets, train_box_per_dataset, test_box_per_dataset, gt_data_pairs, box_vol, box_int, thresh):
    y_pred, y_true = [], []
    for test_d in test_datasets:
        for train_d in train_datasets:
            c_rows = gt_data_pairs[((gt_data_pairs["child"]==test_d)&(gt_data_pairs["parent"]==train_d))]

            # Set train dataset as parent. test dataset as child
            box_p, box_c = train_box_per_dataset[train_d], test_box_per_dataset[test_d]
            overlap_pc = torch.exp(box_vol(box_int(box_p, box_c)) - box_vol(box_p))
            overlap_cp = torch.exp(box_vol(box_int(box_c, box_p)) - box_vol(box_c))
            pred = 0
            if (overlap_cp > overlap_pc) and (overlap_cp > thresh): # child overlap should be higher
                pred = 1
            label = 0
            if len(c_rows) > 0: # means test_d is the child or train_d and overlap should reflect that
                label = 1
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

def evaluate(model, x_train, x_test, train_datasets, test_datasets, gt_data_pairs, box_vol, box_int, thresh=None):
     with torch.no_grad():
        ll_coords, box_sizes = model(x_train)
        ur_coords = ll_coords + box_sizes
        train_boxes = torch.stack((ll_coords, ur_coords), dim=1)
        train_box_per_dataset = {}
        for d_idx, d in enumerate(train_datasets):
            train_box_per_dataset[d] = BoxTensor(train_boxes[d_idx, :, :])

        ll_coords, box_sizes = model(x_test)
        ur_coords = ll_coords + box_sizes
        test_boxes = torch.stack((ll_coords, ur_coords), dim=1)
        test_box_per_dataset = {}
        for d_idx, d in enumerate(test_datasets):
            test_box_per_dataset[d] = BoxTensor(test_boxes[d_idx, :, :])

        if thresh is None:  # means need to find threshold (should use validation set)
            threshes = [x/10. for x in range(1,10)]
            logger.debug(f"Threshold not available. Finding optimal among {threshes}")
            val_f1s, val_precs, val_recs = [], [], []
            for t in threshes:
                val_f1, val_prec, val_rec = get_embed_metrics(
                    train_datasets, test_datasets, train_box_per_dataset, test_box_per_dataset, gt_data_pairs, box_vol, box_int, t
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
            test_f1, test_prec, test_rec = get_embed_metrics(
                train_datasets, test_datasets, train_box_per_dataset, test_box_per_dataset, gt_data_pairs, box_vol, box_int, thresh
            )
            logger.debug(f"Evaluated with threshold={thresh}. f1: {test_f1}, prec: {test_prec}, rec: {test_rec}")
            return test_f1, test_prec, test_rec

def is_only_one_species(all_one_species_datasets, dataset):
    if dataset in all_one_species_datasets:
        return True
    return False

def is_fim_available(fim_dir, dataset):
    for src in ["inat", "cub"]:
        fp = os.path.join(fim_dir, f"{src}_{dataset}.npy")
        if os.path.exists(fp):
            return True
    return False

def get_test_val_datasets(train_datasets, gt_data_pairs, all_one_species_datasets):
    test_val_orders = []
    test_val_families = []
    for d in train_datasets:
        p_rows = gt_data_pairs[(gt_data_pairs["parent"]==d)]
        if len(p_rows) > 0:
            orders = np.unique(p_rows[p_rows["child_type"]=="order"]["child"].values)
            for o in orders:
                if (not is_fim_available(args.fim_dir, o)) or is_only_one_species(all_one_species_datasets, o):
                    continue
                if (o not in train_datasets) and (o not in test_val_orders) and (o not in test_val_families):
                    test_val_orders.append(o)
            families = np.unique(p_rows[p_rows["child_type"]=="family"]["child"].values)
            for f in families:
                if (not is_fim_available(args.fim_dir, f)) or is_only_one_species(all_one_species_datasets, f):
                    continue
                if (f not in train_datasets) and (f not in test_val_families) and (f not in test_val_orders):
                    test_val_families.append(f)
    random.shuffle(test_val_orders)
    random.shuffle(test_val_families)

    test_val_datasets = test_val_orders+test_val_families
    val_datasets = test_val_datasets[:NUM_VAL_DATASETS]
    test_datasets = test_val_datasets[NUM_VAL_DATASETS:(NUM_VAL_DATASETS+NUM_TEST_DATASETS)]
    return test_datasets, val_datasets

def get_input_test_cubinat(fim_dir, gauss_fp, test_datasets):
    with open(gauss_fp, "rb") as f:
        gauss_data = pickle.load(f)
    in_feats = {
        "clip_gauss": [],
        "fim": [],
        "clip_ave": [],
    }
    for ord_fam in test_datasets:
        mean, var, num_samps = gauss_data["gauss_params"][ord_fam]
        gauss_feat = np.stack((mean, var), axis=0)
        in_feats["clip_gauss"].append(np.reshape(gauss_feat, (1,-1)))
        in_feats["clip_ave"].append(np.reshape(mean, (1,-1)))

        for src in ["inat", "cub"]:
            fp = os.path.join(fim_dir, f"{src}_{ord_fam}.npy")  #embeddings/task2vec/inat_Phoenicopteriformes.npy
            if not os.path.exists(fp):
                continue
            with open(fp, "rb") as f:
                fim_data = np.load(fp, allow_pickle=True)
                in_feats["fim"].append(np.reshape(fim_data, (1,-1)))
                break
    for k,v in in_feats.items():
        in_feats[k] = np.concatenate(in_feats[k], axis=0)
        # print(k, in_feats[k].shape)
        assert in_feats[k].shape[0] == len(test_datasets)
    
    return in_feats

def get_input_test_imagenet(fim_dir, gauss_fp, test_datasets):
    with open(gauss_fp, "rb") as f:
        gauss_data = pickle.load(f)
    in_feats = {
        "clip_gauss": [],
        "fim": [],
        "clip_ave": [],
    }
    for d_name in test_datasets:
        # logger.debug(f"Getting embeddings for {d_name}")
        mean, var, num_samps = gauss_data["gauss_params"][d_name]
        gauss_feat = np.stack((mean, var), axis=0)
        in_feats["clip_gauss"].append(np.reshape(gauss_feat, (1,-1)))
        in_feats["clip_ave"].append(np.reshape(mean, (1,-1)))

        fp = os.path.join(fim_dir, f"imagenet_{d_name}.npy")  #embeddings/task2vec/inat_Phoenicopteriformes.npy
        with open(fp, "rb") as f:
            fim_data = np.load(fp, allow_pickle=True)
            in_feats["fim"].append(np.reshape(fim_data, (1,-1)))
    for k,v in in_feats.items():
        in_feats[k] = np.concatenate(in_feats[k], axis=0)
        # print(k, in_feats[k].shape)
        assert in_feats[k].shape[0] == len(test_datasets)
    
    return in_feats

def get_metric_for_random_links(train_datasets, test_datasets, gt_data_pairs, p=[0.5, 0.5]):
    y_pred, y_true = [], []
    for test_d in test_datasets:
        for train_d in train_datasets:
            c_rows = gt_data_pairs[((gt_data_pairs["child"]==test_d)&(gt_data_pairs["parent"]==train_d))]

            label = 0
            if len(c_rows) > 0: # means test_d is the child or train_d and overlap should reflect that
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

def get_prob_1s(train_datasets, test_datasets, gt_data_pairs):
    num_neg = 0
    num_pos = 0
    for test_d in test_datasets:
        for train_d in train_datasets:
            c_rows = gt_data_pairs[((gt_data_pairs["child"]==test_d)&(gt_data_pairs["parent"]==train_d))]

            label = 0
            if len(c_rows) > 0: # means test_d is the child or train_d and overlap should reflect that
                label = 1

            if label == 0:
                num_neg += 1
            else:
                num_pos += 1
    return num_pos/(num_pos + num_neg)

def linear_evaluate(model, val_loader, thresh=None):
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

def get_embed_label_from_pair_idxs(gt_data_pairs, train_datasets, test_datasets):
    labels = [] # NOTE: MLP/Linear models are is-parent(x,y) models that predicts if x is a parent of y
    train_test_idxs = []
    for train_idx, train_d in enumerate(train_datasets):
        for test_idx, test_d in enumerate(test_datasets):
            relevant_rows = gt_data_pairs[((gt_data_pairs["parent"]==train_d)&(gt_data_pairs["child"]==test_d))]
            label = 0
            if len(relevant_rows) > 0:
                label = 1
            labels.append(label)
            train_test_idxs.append((train_idx, test_idx))
    assert len(labels) == len(train_test_idxs)
    return labels, train_test_idxs

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

    if args.dataset == "cubinat":
        data_fp = f"../embeddings/{args.link_pred_src}/case{args.case_num:02d}.pickle"
    elif args.dataset == "imagenet":
        data_fp = f"../embeddings/{args.link_pred_src}/imagenet_case{args.case_num:02d}.pickle"
    else:
        raise NotImplementedError
    with open(data_fp, "rb") as f:
        all_data = pickle.load(f)

    # Load datasets
    if args.dataset == "cubinat":
        # Remove species that only have 1 sample or have disagreements between cub/inat hierarchy
        filenames = ["cub-one-species-family.txt", "cub-one-species-order.txt", "inat-one-species-class.txt","inat-one-species-family.txt", "inat-one-species-order.txt", "disagree-taxonomy.txt"]
        all_one_species_datasets = []
        for fn in filenames:
            fp = os.path.join(args.txt_labels_dir, fn)
            with open(fp) as f:
                lines = f.readlines()
            assert len(lines)==1
            line = lines[0]
            one_species_datasets = [d.strip() for d in line.split(" ")]
            all_one_species_datasets += one_species_datasets

        train_datasets = all_data["datasets"]
        test_datasets, val_datasets = get_test_val_datasets(train_datasets, gt_data_pairs, all_one_species_datasets)

        # Load input
        in_feats_val = get_input_test_cubinat(args.fim_dir, args.gauss_fp, val_datasets)
        in_feats_test = get_input_test_cubinat(args.fim_dir, args.gauss_fp, test_datasets)
    elif args.dataset == "imagenet":
        train_datasets = all_data["train_datasets"]
        test_datasets = all_data["test_datasets"] 
        val_datasets = all_data["val_datasets"]

        # Load input
        in_feats_val = get_input_test_imagenet(args.fim_dir, args.gauss_fp, val_datasets)
        in_feats_test = get_input_test_imagenet(args.fim_dir, args.gauss_fp, test_datasets)
    
    val_labels, train_val_idxs = get_embed_label_from_pair_idxs(gt_data_pairs, train_datasets, val_datasets)
    test_labels, train_test_idxs = get_embed_label_from_pair_idxs(gt_data_pairs, train_datasets, test_datasets)

    if args.model_type == "random":
        test_f1, test_prec, test_rec = get_metric_for_random_links(train_datasets, test_datasets, gt_data_pairs, p=[0.5, 0.5])
        logger.debug(f"random case {args.case_num:02d} test_f1: {test_f1}, test_prec: {test_prec}, test_rec: {test_rec}")
        all_metrics.append([args.model_type, None, None, args.case_num, test_f1, test_prec, test_rec, None, args.link_pred_src.split("new")[-1], None])
        df = pd.DataFrame(all_metrics, columns=["model", "feat_type", "box_dim", "case_num", "f1", "prec", "rec", "thresh", "supervision_perc", "epoch", "ckpt_fp"])
        df.to_csv(f"{args.results_dir}/{args.model_type}_{args.feat_type}_{args.box_dim}_{args.case_num}_embed_pred_metrics_{args.link_pred_src}.csv")
        exit()
    elif args.model_type == "weighted_random":
        prob_one = get_prob_1s(train_datasets, test_datasets, gt_data_pairs)
        prob_zero = 1-prob_one
        logger.debug(f"weighted_random prob_one: {prob_one}, prob_zero: {prob_zero}")
        test_f1, test_prec, test_rec = get_metric_for_random_links(train_datasets, test_datasets, gt_data_pairs, p=[prob_zero, prob_one])
        logger.debug(f"weighted_random case {args.case_num:02d} test_f1: {test_f1}, test_prec: {test_prec}, test_rec: {test_rec}")
        all_metrics.append([args.model_type, None, None, args.case_num, test_f1, test_prec, test_rec, None, args.link_pred_src.split("new")[-1], None])
        df = pd.DataFrame(all_metrics, columns=["model", "feat_type", "box_dim", "case_num", "f1", "prec", "rec", "thresh", "supervision_perc", "epoch", "ckpt_fp"])
        df.to_csv(f"{args.results_dir}/{args.model_type}_{args.feat_type}_{args.box_dim}_{args.case_num}_embed_pred_metrics_{args.link_pred_src}.csv")
        exit()

    epoch = args.epoch

    in_feats = all_data[f"{args.feat_type}_train_data"]
    x_train = torch.from_numpy(in_feats).type(torch.FloatTensor)
    x_val = torch.from_numpy(in_feats_val[args.feat_type]).type(torch.FloatTensor)
    x_test = torch.from_numpy(in_feats_test[args.feat_type]).type(torch.FloatTensor)


    # Load checkpoint
    if args.model_type == "Task2Box":
        all_ckpts = glob.glob(os.path.join(
            args.ckpts_dir, f'*_{args.feat_type}_case{args.case_num:02d}_{epoch:02d}_box{args.box_dim:02d}_model.pth'
        ))
    else:
        all_ckpts = glob.glob(os.path.join(
            args.ckpts_dir, f'*_{args.model_type}_{args.feat_type}_case{args.case_num:02d}_{epoch:02d}_model.pth'
        ))

    all_ckpts = sorted(all_ckpts)
    logger.debug(f"Looking for checkpoints with pattern: *_{args.feat_type}_case{args.case_num:02d}_{epoch:02d}_box{args.box_dim:02d}_model.pth")
    logger.debug(f"all_ckpts: {all_ckpts}")
    if not (len(all_ckpts) >= 1):
        for e in list(range(1500,0,-75)):
            if args.model_type == "Task2Box":
                new_pattern = f'*_{args.feat_type}_case{args.case_num:02d}_{e:02d}_box{args.box_dim:02d}_model.pth'
                all_ckpts = glob.glob(os.path.join(args.ckpts_dir, new_pattern))
            else:
                new_pattern = f'*_{args.model_type}_{args.feat_type}_case{args.case_num:02d}_{e:02d}_model.pth'
                all_ckpts = glob.glob(os.path.join(args.ckpts_dir, new_pattern))

            logger.error(f"Checkpoint not found. Using new pattern: {new_pattern}")
            all_ckpts = sorted(all_ckpts)
            if len(all_ckpts) >= 1:
                # epoch = all_ckpts[-1].split("_")[-2]
                epoch = e
                logger.warning(f"Using epoch={epoch} instead. Choices: {all_ckpts}")
                break
        else:
            logger.error("No alternative checkpoints found. Exiting")
            exit()
    for ckpt_fp in all_ckpts:
        # ckpt_fp = all_ckpts[-1] # Get most recent checkpoint
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
            model.load_state_dict(ckpt["model_state_dict"], strict=True)

            # Run model on input embeddings
            model.eval()
            val_f1s, val_precs, val_recs, optim_thresh = evaluate(
                model, x_train, x_val, train_datasets, val_datasets, gt_data_pairs, box_vol, box_int, thresh=None
            )
            logger.debug(f"Optim thresh={optim_thresh}. All f1s: {val_f1s}. val_precs: {val_precs}. val_recs: {val_recs}")
            test_f1, test_prec, test_rec= evaluate(
                model, x_train, x_test, train_datasets, test_datasets, gt_data_pairs, box_vol, box_int, thresh=optim_thresh
            )
        elif args.model_type == "mlp":
            args.box_dim = None
            model = MLPModel(in_dim=in_dim*2)
            model.load_state_dict(ckpt["model_state_dict"], strict=True)

            val_dataset = EmbedPairData(in_feats, in_feats_val[args.feat_type], train_val_idxs, val_labels)
            val_loader = torch.utils.data.DataLoader(
                dataset=val_dataset,
                batch_size=args.batch_size,
                shuffle=False, num_workers=0)
            test_dataset = EmbedPairData(in_feats, in_feats_test[args.feat_type], train_test_idxs, test_labels)
            test_loader = torch.utils.data.DataLoader(
                dataset=test_dataset,
                batch_size=args.batch_size,
                shuffle=False, num_workers=0)
            
            model.eval()
            optim_thresh, val_f1s, val_precs, val_recs = linear_evaluate(model, val_loader, thresh=None)
            test_f1, test_prec, test_rec = linear_evaluate(model, test_loader, thresh=optim_thresh)

        elif args.model_type == "linear":
            args.box_dim = None
            model = LinearModel(in_dim=in_dim*2)
            model.load_state_dict(ckpt["model_state_dict"], strict=True)

            val_dataset = EmbedPairData(in_feats, in_feats_val[args.feat_type], train_val_idxs, val_labels)
            val_loader = torch.utils.data.DataLoader(
                dataset=val_dataset,
                batch_size=args.batch_size,
                shuffle=False, num_workers=0)
            test_dataset = EmbedPairData(in_feats, in_feats_test[args.feat_type], train_test_idxs, test_labels)
            test_loader = torch.utils.data.DataLoader(
                dataset=test_dataset,
                batch_size=args.batch_size,
                shuffle=False, num_workers=0)

            model.eval()
            optim_thresh, val_f1s, val_precs, val_recs = linear_evaluate(model, val_loader, thresh=None)
            test_f1, test_prec, test_rec = linear_evaluate(model, test_loader, thresh=optim_thresh)
        else:
            raise NotImplementedError
        all_metrics.append([args.model_type, args.feat_type, args.box_dim, args.case_num, test_f1, test_prec, test_rec, optim_thresh, args.link_pred_src.split("new")[-1], epoch, ckpt_fp])
    
        df = pd.DataFrame(all_metrics, columns=["model", "feat_type", "box_dim", "case_num", "f1", "prec", "rec", "thresh", "supervision_perc", "epoch", "ckpt_fp"])
        
        df.to_csv(f"{args.results_dir}/{args.model_type}_{args.feat_type}_{args.box_dim}_{args.case_num}_embed_pred_metrics_{args.link_pred_src}.csv")