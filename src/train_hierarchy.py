import torch
import torch.nn as nn
import numpy as np
from box_embeddings.parameterizations.box_tensor import BoxTensor
from box_embeddings.modules.volume.volume import Volume
from box_embeddings.modules.intersection import Intersection

import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from loguru import logger
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import random
import argparse
import pickle
import json
import os

from utils.model import BoxModel

EPS = 1e-12
DATE_STR = datetime.now().strftime("%Y%m%d-%H%M%S")
parser = argparse.ArgumentParser(description='Task2Box training script')
parser.add_argument('--dataset', default="cubinat", type=str,
                    help='Dataset to use (cubinat, imagenet)')
parser.add_argument('--case_num', default=1, type=int,
                    help='Case number for training/evaluation [1-10] ')
parser.add_argument('--link_pred_src', default="link_pred_new50", type=str,
                    help='Where to get train data')
parser.add_argument('--gt_pairs_fp', default="../data/hierarchy_both.csv", type=str,
                    help='Directory for saving results')
parser.add_argument('--ckpts_dir', default="../ckpts", type=str,
                    help='Directory where dataset cache is downloaded/stored')
parser.add_argument('--batch_size', default=2, type=int,
                    help='batch size. ')
parser.add_argument('--feat_type', default="clip_gauss", type=str,
                    help='Model used for extracting embeddings ("clip_ave", "clip_gauss", "fim")')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. Default is None but can set to number 123 (orig)')
parser.add_argument('--num_epochs', default=1350, type=int,
                    help='Number of epochs for training box embeddings')


parser.add_argument('--oloss_mult', default=100, type=int,
                    help='Multiplier of overlap loss')
parser.add_argument('--rloss_mult', default=1, type=float,
                    help='Multiplier of reg loss')
parser.add_argument('--dloss_mult', default=0.01, type=float,
                    help='Multiplier of distance loss')
parser.add_argument('--vloss_mult', default=0.7, type=float,
                    help='Multiplier of volume loss')


parser.add_argument('--lr', default=0.001, type=float,
                    help='Learning rate for optimizer')
parser.add_argument('--optimizer', default="adam", type=str,
                    help='Type of optimizer to use ("adam", "sgd")')
parser.add_argument('--box_dim', default=2, type=int,
                    help='Dimension of box embedding (2 for visualization -- will add viz for 3 later)')
parser.add_argument('--is_gradient_clip', default=0, type=int,
                    help='1 to do gradient clipping. 0 to NOT do gradient clipping')


parser.add_argument('--pct_saving_interval', default=50, type=int,
                    help='Percent of num_epochs as saving interval for checkpoints (0-100)')



def plot_losses(losses, epoch, date_str, args, plot_type="losses", start_epoch=0):
    plt.figure()
    if plot_type == "losses":
        plt.yscale("log")
    epochs = losses.get("epoch") or None
    if epochs is None:
        epochs = [i+1 for i in range(start_epoch, epoch+1)]
    for k, v in losses.items():
        if k == "epoch":    # don't plot epochs
            continue
        if v:
            plt.plot(epochs, v, label = k)
    plt.legend()
    plt.savefig(f"{args.ckpts_dir}/{date_str}_case{args.case_num:02d}_{args.feat_type}_box{args.box_dim:02d}_{plot_type}.jpg", bbox_inches='tight')
    plt.close()


def display_boxes(Y_outs, datasets, date_str, args, epoch=0):
    if args.box_dim != 2:
        logger.warning(f"Cannot display box since box_dim={args.box_dim}")
        return
    fig, ax = plt.subplots()
    colors = list(mcolors.CSS4_COLORS.keys())+list(mcolors.CSS4_COLORS.keys())
    for idx, y in enumerate(Y_outs):
        height = y[1][1] - y[0][1]
        width = y[1][0] - y[0][0]
        rect = patches.Rectangle(y[0], width=width, height=height, facecolor="none", edgecolor=colors[idx], label=datasets[idx])

        ax.add_patch(rect)
    plt.ylim(np.min(np.array(Y_outs)[:,1])-3,np.max(np.array(Y_outs)[:,1])+3)
    plt.xlim(np.min(np.array(Y_outs)[:,0])-3,np.max(np.array(Y_outs)[:,0])+3)
    plt.savefig(f"{args.ckpts_dir}/{date_str}_case{args.case_num:02d}_{args.feat_type}_{epoch:02d}_boxes{args.box_dim:02d}.png")
    plt.close()


def train(args, model, x, idx_perm, train_idxs_flat, optimizer, D_sim_train, box_vol, box_int):
    overlap_loss_fn = nn.MSELoss()
    loss = 0
    optimizer.zero_grad()
    
    ll_coords, box_sizes = model(x)
    ur_coords = ll_coords + box_sizes
    boxes = torch.stack((ll_coords, ur_coords), dim=1)  # shape: (num_boxes, 2, box_dim)

    # Reverse permuation of boxes
    boxes_orig_idx = torch.zeros_like(boxes)
    for ctr, real_idx in enumerate(idx_perm):
        boxes_orig_idx[real_idx,:,:] = boxes[ctr,:,:]
        
    all_box_tensors = BoxTensor(boxes_orig_idx)

    # loss to encourage nonzero volume
    vol_loss = 1 / (torch.exp(box_vol(all_box_tensors)) + EPS)
    total_vol_loss = torch.sum((vol_loss)**(1/args.box_dim))

    # Redefine boxes to compute pairwise losses (overlap and dist)
    boxes1 = boxes_orig_idx.unsqueeze(1).repeat(1,len(boxes),1,1).reshape(-1,2,args.box_dim)
    boxes1 = BoxTensor(boxes1)
    boxes2 = boxes_orig_idx.repeat(len(boxes),1,1)
    boxes2 = BoxTensor(boxes2)

    # Compute overlap loss
    overlap_pred = torch.exp(box_vol(box_int(boxes1, boxes2)) - box_vol(boxes1))
    overlap_pred = overlap_pred[train_idxs_flat]
    tot_overlap_loss = overlap_loss_fn(overlap_pred, D_sim_train)*len(D_sim_train)

    # Loss to encourage overlapping boxes to move close to each other 
    b1_center = torch.sum(boxes1.data, axis=1)/2.
    b2_center = torch.sum(boxes2.data, axis=1)/2.
    dist_sq = torch.sum((b1_center - b2_center)**2, axis=1)
    dist_sq = dist_sq[train_idxs_flat]
    relevant_d_idxs = np.where(D_sim_train>EPS)[0]    # only encourage boxes that have overlap to move closer to each other
    dist_sq = dist_sq[relevant_d_idxs]
    total_dist_loss = torch.sum(dist_sq)

    # Compute box regularization loss (to encourage "square" shapes)
    box_reg = 0
    for dim in range(0, box_sizes.shape[1]):
        for dim2 in range(dim+1, box_sizes.shape[1]):
            box_reg += torch.sum(torch.abs(box_sizes[:,dim]-box_sizes[:,dim2]))

    tot_overlap_loss = args.oloss_mult*tot_overlap_loss
    loss += tot_overlap_loss
    
    total_vol_loss = args.vloss_mult*total_vol_loss
    loss += total_vol_loss

    total_dist_loss = args.dloss_mult*total_dist_loss
    loss += total_dist_loss

    box_reg = args.rloss_mult*box_reg*(1/(args.box_dim**2))
    loss += box_reg

    loss.backward()
    if args.is_gradient_clip:
        torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)
    optimizer.step()

    return loss, tot_overlap_loss, total_dist_loss, total_vol_loss, box_reg, boxes

def evaluate(args, model, x, datasets, mask_idxs, filtered_gt_pairs, box_vol, box_int, thresh=None):
    model.eval()
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
            optimal_idx = np.nanargmax(val_f1s)
            optimal_threshold = threshes[optimal_idx]
            logger.debug(f"Found optimal threshold: {optimal_threshold}. f1 score: {val_f1s[optimal_idx]}. Candidates: {val_f1s}")
            return optimal_threshold
        else:
            test_f1, test_prec, test_rec = get_metrics(
                mask_idxs, box_per_dataset, datasets, filtered_gt_pairs, box_vol, box_int, thresh
            )
            logger.debug(f"Evaluated with threshold={thresh}. f1: {test_f1}, prec: {test_prec}, rec: {test_rec}")
            return test_f1, test_prec, test_rec


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
    recall = TP/(TP+FN)
    precision = TP/(TP+FP)
    f1 = (2*precision*recall)/(precision+recall)

    return f1, precision, recall

if __name__=="__main__":
    args = parser.parse_args()
    logger.debug(f"args: {args}")
    if not os.path.exists(args.ckpts_dir):
        os.makedirs(args.ckpts_dir, exist_ok=True)

    tmp_args = vars(args)
    with open(f"{args.ckpts_dir}/{DATE_STR}_{args.feat_type}_case{args.case_num:02d}_{args.feat_type}_boxes{args.box_dim:02d}__params.json", "w") as f:
        json.dump(tmp_args, f, indent=4)
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Load input features for each dataset/task (e.g., for 100 tasks x_init should have shape of (100,in_dim))
    if args.dataset == "cubinat":
        data_fp = f"../embeddings/{args.link_pred_src}/case{args.case_num:02d}.pickle"
    elif args.dataset == "imagenet":
        data_fp = f"../embeddings/{args.link_pred_src}/imagenet_case{args.case_num:02d}.pickle"
    else:
        raise NotImplementedError
    with open(data_fp, "rb") as f:
        all_data = pickle.load(f)
    mask = all_data["mask"]
    train_mask = (mask==0).astype(int)
    val_mask = (mask==1).astype(int)
    test_mask = (mask==2).astype(int)

    tmp_train_mask = np.where(train_mask>0)
    train_mask_idxs = list(zip(tmp_train_mask[0], tmp_train_mask[1]))
    tmp_val_mask = np.where(val_mask>0)
    val_mask_idxs = list(zip(tmp_val_mask[0], tmp_val_mask[1]))
    tmp_test_mask = np.where(test_mask>0)
    test_mask_idxs = list(zip(tmp_test_mask[0], tmp_test_mask[1]))

    if args.dataset == "cubinat":
        datasets_train = all_data["datasets"]
    elif args.dataset == "imagenet":
        datasets_train = all_data["train_datasets"]
    dists_key_name = f"{args.feat_type}_kl_dists" if args.feat_type!="clip_ave" else "clip_ave_euc_dists"
    train_overlaps = all_data["overlap"]

    data_key_name = f"{args.feat_type}_train_data"
    in_feats = all_data[data_key_name]
    in_dim = in_feats.shape[1]
    x_init = torch.from_numpy(in_feats).type(torch.FloatTensor)  # shape: (num_tasks, in_dim)    # fixed input features


    # The following is for evaluation
    gt_data_pairs = pd.read_csv(args.gt_pairs_fp, index_col=0)
    gt_data_pairs.dropna(inplace=True)
    filtered_gt_pairs = gt_data_pairs[((gt_data_pairs["parent"].isin(datasets_train))|(gt_data_pairs["child"].isin(datasets_train)))]
    logger.debug(f"Loaded GT pairs shape: {gt_data_pairs.shape}. Filtered to: {len(filtered_gt_pairs)} pairs")


    model = BoxModel(box_dim=args.box_dim, in_dim=in_dim)

    D_similarity = np.maximum(train_overlaps, EPS)
    D_overlaps = torch.from_numpy(D_similarity).type(torch.FloatTensor)
    
    if args.optimizer == "sgd":
        optimizer =  torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == "adam":
        optimizer =  torch.optim.Adam(model.parameters(), lr=args.lr)

    box_vol = Volume(volume_temperature=0.1, intersection_temperature=0.0001)
    box_int = Intersection(intersection_temperature=0.0001)

    train_mask_tmp = np.zeros_like(train_mask)
    train_mask_tmp[:,:] = train_mask[:,:]
    np.fill_diagonal(train_mask_tmp, 0)    # in overlap and dist loss, don't include overlap of box with itself
    train_mask_flat = np.reshape(train_mask_tmp, (-1,))
    train_idxs_flat = np.where(train_mask_flat>0)[0]
    D_sim_train = D_overlaps.reshape(-1,)[train_idxs_flat]
    
    losses = {"train_loss": [], "overlap_loss": [], "box_reg": [], "dist_loss": [], "vol_loss": []}
    metrics = { "epoch": [], "val_f1": [],"train_f1": [], "val_prec": [],"train_prec": [], "val_rec": [], "train_rec": []}
    for epoch in tqdm(range(args.num_epochs)):
        idx_perm = torch.randperm(x_init.shape[0])  # shuffle train data
        x_train = x_init[idx_perm]

        model.train()
        train_loss, overlap_loss, dist_loss, vol_loss, box_reg, out_boxes = train(
            args,
            model,
            x_train,
            idx_perm,
            train_idxs_flat,
            optimizer,
            D_sim_train,
            box_vol,
            box_int,
        )
        
        losses["train_loss"].append(train_loss.item())
        losses["overlap_loss"].append(overlap_loss.item())
        losses["dist_loss"].append(dist_loss.item())
        losses["vol_loss"].append(vol_loss.item())
        if box_reg:
            losses["box_reg"].append(box_reg.item())
        else:
            losses["box_reg"].append(0)
        plot_losses(losses, epoch, DATE_STR, args)

        # Visualize boxes and save model checkpoint
        if epoch%(int(args.num_epochs*args.pct_saving_interval/100))==0:
            logger.debug(f"train_loss: {train_loss}, overlap_loss: {overlap_loss}, dist_loss: {dist_loss}, vol_loss: {vol_loss}, box_reg: {box_reg}")
            display_boxes(out_boxes.detach().cpu().numpy(), list(range(len(x_init))), DATE_STR, args, epoch)
            
            logger.debug("Validation set evaluation")
            val_f1, val_prec, val_rec = evaluate(
                args, model, x_init, datasets_train, val_mask_idxs, filtered_gt_pairs, box_vol, box_int, thresh=0.5
            )   # NOTE: using 0.5 as threshold for intermediate validation
            metrics["epoch"].append(epoch)
            metrics["val_f1"].append(val_f1)
            metrics["val_prec"].append(val_prec)
            metrics["val_rec"].append(val_rec)

            model_ckpt_fp = os.path.join(args.ckpts_dir, f"{DATE_STR}_{args.feat_type}_case{args.case_num:02d}_{epoch:02d}_box{args.box_dim:02d}_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses': losses,
                'metrics': metrics,
                }, model_ckpt_fp)
            plot_losses(metrics, epoch, DATE_STR, args, plot_type="metrics")
            
    # Save for final epoch
    logger.debug(f"train_loss: {train_loss}, overlap_loss: {overlap_loss}, dist_loss: {dist_loss}, vol_loss: {vol_loss}, box_reg: {box_reg}")
    display_boxes(out_boxes.detach().cpu().numpy(), list(range(len(x_init))), DATE_STR, args, epoch)
    logger.debug("Validation set evaluation")
    val_f1, val_prec, val_rec = evaluate(
        args, model, x_init, datasets_train, val_mask_idxs, filtered_gt_pairs, box_vol, box_int, thresh=0.5
    )   # NOTE: using 0.5 as threshold for intermediate validation
    metrics["epoch"].append(epoch)
    metrics["val_f1"].append(val_f1)
    metrics["val_prec"].append(val_prec)
    metrics["val_rec"].append(val_rec)

    logger.debug("Train set evaluation")
    plot_losses(metrics, epoch, DATE_STR, args, plot_type="metrics")
    
    optim_thresh = evaluate(
        args, model, x_init, datasets_train, val_mask_idxs, filtered_gt_pairs, box_vol, box_int, thresh=None
    )
    f1, prec, rec = evaluate(
        args, model, x_init, datasets_train, test_mask_idxs, filtered_gt_pairs, box_vol, box_int, thresh=optim_thresh
    )
    model_ckpt_fp = os.path.join(args.ckpts_dir, f"{DATE_STR}_{args.feat_type}_case{args.case_num:02d}_{epoch:02d}_box{args.box_dim:02d}_model.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'metrics': metrics,
        "test_metrics": [f1, prec, rec, optim_thresh],
    }, model_ckpt_fp)