from box_embeddings.parameterizations.box_tensor import *
from box_embeddings.modules.volume.soft_volume import soft_volume

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
import scipy.stats as stats

from utils.model import BoxModel, BoxModelSmall, BoxModelLinear
from utils.model import MLPModel, LinearModel
from utils.dataloader import TaskonomyPairData

EPS = 1e-12
DATE_STR = datetime.now().strftime("%Y%m%d-%H%M%S")
parser = argparse.ArgumentParser(description='Image-label embedding')
parser.add_argument('--case_num', default=1, type=int,
                    help='Case number for training/evaluation [1-10] ')
parser.add_argument('--link_pred_src', default="link_pred_new80", type=str,
                    help='Where to get train data')
parser.add_argument('--ckpts_dir', default="../ckpts_taskonomy_50", type=str,
                    help='Directory where checkpoints from training are saved')
parser.add_argument('--batch_size', default=16, type=int,
                    help='batch size. ')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. (None by default. Can set to specific number (123 - orig))')
parser.add_argument('--num_epochs', default=3000, type=int,
                    help='Number of epochs for training box embeddings')


parser.add_argument('--tanh_k', default=50, type=int,
                    help='Constant used to map overlap to tanh')
parser.add_argument('--model_type', default="Task2Box", type=str,
                    help='Model to use for training. Choices: ["Task2Box", "Task2BoxSmall", "Task2BoxLinear", "linear", "mlp"]')
parser.add_argument('--loss_type', default="mse", type=str,
                    help='Model to use for training. Choices: ["mse","l1"]')


parser.add_argument('--expk', default=2, type=int,
                    help='Multiplier of KL-div for overlap proxy value')
parser.add_argument('--oloss_mult', default=100, type=int,
                    help='Multiplier of overlap loss')
parser.add_argument('--rloss_mult', default=1, type=float,
                    help='Multiplier of reg loss')
parser.add_argument('--dloss_mult', default=0.01, type=float,
                    help='Multiplier of distance loss')
parser.add_argument('--vloss_mult', default=0.01, type=float,
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
    plt.savefig(f"{args.ckpts_dir}/{date_str}_case{args.case_num:02d}_taskonomy_box{args.box_dim:02d}_{plot_type}.jpg", bbox_inches='tight')
    plt.close()


def display_boxes(Y_outs, datasets, date_str, args, epoch=0):
    if args.box_dim != 2:
        logger.warning(f"Cannot display box since box_dim={args.box_dim}")
        return
    fig, ax = plt.subplots()
    # colors = ["r", "g", "b", "y", "gray"]
    colors = list(mcolors.CSS4_COLORS.keys())+list(mcolors.CSS4_COLORS.keys())
    for idx, y in enumerate(Y_outs):
        height = y[1][1] - y[0][1]
        width = y[1][0] - y[0][0]
        rect = patches.Rectangle(y[0], width=width, height=height, facecolor="none", edgecolor=colors[idx], label=datasets[idx])

        ax.add_patch(rect)
    plt.ylim(np.min(np.array(Y_outs)[:,1])-3,np.max(np.array(Y_outs)[:,1])+3)
    plt.xlim(np.min(np.array(Y_outs)[:,0])-3,np.max(np.array(Y_outs)[:,0])+3)
    plt.legend()
    # plt.show()
    plt.savefig(f"{args.ckpts_dir}/{date_str}_case{args.case_num:02d}_taskonomy_{epoch:02d}_boxes{args.box_dim:02d}.png")
    plt.close()

def train_linear(args, model, train_loader, optimizer):
    loss_fn = nn.MSELoss()
    model.train()
    for x,y in train_loader:
        if x.shape[0]==1:   # skip batches with just 1 member
            continue
        y = y.type(torch.FloatTensor)
        x = x.type(torch.FloatTensor)
        optimizer.zero_grad()
        preds = model(x)
        loss = loss_fn(
            torch.reshape(preds, (-1,)),
            torch.reshape(y, (-1,))
        )
        loss.backward()
        if args.is_gradient_clip:
            torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)
        optimizer.step()
    return loss



def train(args, model, x, idx_perm, train_idxs_flat, optimizer, D_sim_train, box_vol, box_int):
    if args.loss_type == "mse":
        overlap_loss_fn = nn.MSELoss()
    elif args.loss_type == "l1":
        overlap_loss_fn = nn.L1Loss()
    loss = 0
    optimizer.zero_grad()
    
    ll_coords, box_sizes = model(x)
    ur_coords = ll_coords + box_sizes
    boxes = torch.stack((ll_coords, ur_coords), dim=1)

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


def evaluate_on_new_datasets(args, model, x_train, x_val, train_datasets, val_datasets, box_vol, box_int, full_overlaps_transf, full_overlaps):
    model.eval()
    with torch.no_grad():
        if args.model_type in ["Task2Box", "Task2BoxSmall", "Task2BoxLinear"]:
            ll_coords, box_sizes = model(x_train)
            ur_coords = ll_coords + box_sizes
            train_boxes = torch.stack((ll_coords, ur_coords), dim=1)

            train_box_per_dataset = {}
            for d_idx, d in enumerate(train_datasets):
                train_box_per_dataset[d] = BoxTensor(train_boxes[d_idx, :, :])

            val_ll_coords, val_box_sizes = model(x_val)
            val_ur_coords = val_ll_coords + val_box_sizes
            val_boxes = torch.stack((val_ll_coords, val_ur_coords), dim=1)
            val_box_per_dataset = {}
            for d_idx, d in enumerate(val_datasets):
                val_box_per_dataset[d] = BoxTensor(val_boxes[d_idx, :, :])

        orig_preds, preds = [], []
        orig_labels, labels = [], []
        for train_idx, train_d in enumerate(train_datasets):
            for val_idx, val_d in enumerate(val_datasets):
                if args.model_type in ["Task2Box", "Task2BoxSmall", "Task2BoxLinear"]:
                    box1, box2 = train_box_per_dataset[train_d], val_box_per_dataset[val_d]
                    overlap_pred = torch.exp(box_vol(box_int(box1, box2)) - box_vol(box1))
                elif args.model_type in ["linear", "mlp"]:
                    in1 = torch.reshape(x_train[train_idx], (1,-1))
                    in2 = torch.reshape(x_val[val_idx], (1,-1))
                    concat_inp = torch.concat([in1,in2], dim=1)
                    overlap_pred = model(concat_inp)

                orig_label = full_overlaps[train_idx, val_idx]
                label = full_overlaps_transf[train_idx, val_idx]
                if not np.isnan(orig_label):    # only evaluate on non-nan affinity values
                    preds.append(overlap_pred.item())
                    labels.append(label)
                    orig_pred = get_tanh_inv(overlap_pred.item(), k=args.tanh_k)
                    orig_preds.append(orig_pred)
                    orig_labels.append(orig_label)

                # Opposite relationship should also hold
                if args.model_type in ["Task2Box", "Task2BoxSmall", "Task2BoxLinear"]:
                    overlap_pred = torch.exp(box_vol(box_int(box2, box1)) - box_vol(box2))
                elif args.model_type in ["linear", "mlp"]:
                    concat_inp = torch.concat([in2,in1], dim=1)
                    overlap_pred = model(concat_inp)

                label = full_overlaps_transf[val_idx, train_idx]
                orig_label = full_overlaps[val_idx, train_idx]
                if not np.isnan(orig_label):    # only evaluate on non-nan affinity values
                    preds.append(overlap_pred.item())
                    labels.append(label)
                    orig_pred = get_tanh_inv(overlap_pred.item(), k=args.tanh_k)
                    orig_preds.append(orig_pred)
                    orig_labels.append(orig_label)

        orig_preds, preds = np.array(orig_preds), np.array(preds)
        orig_labels, labels = np.array(orig_labels), np.array(labels)
        rho, p_value = stats.spearmanr(labels, preds)
        rho_orig, p_value_orig = stats.spearmanr(orig_labels, orig_preds)
        logger.debug(f"rho: {rho}, p_value: {p_value}. num samples: {len(labels)}")
        logger.debug(f"rho_orig: {rho_orig}, p_value_orig: {p_value_orig}")
    return rho, p_value, rho_orig, p_value_orig, orig_preds, orig_labels


def evaluate(args, model, x_init, val_mask_idxs, box_vol, box_int, transformed_overlaps, orig_overlaps):
    model.eval()
    with torch.no_grad():
        if args.model_type in ["Task2Box", "Task2BoxSmall", "Task2BoxLinear"]:
            ll_coords, box_sizes = model(x_init)
            ur_coords = ll_coords + box_sizes
            boxes = torch.stack((ll_coords, ur_coords), dim=1)

            box_tensors = [BoxTensor(boxes[d_idx, :, :]) for d_idx in range(ll_coords.shape[0])]

        orig_preds, preds = [], []
        orig_labels, labels = [], []
        for idx1, idx2 in val_mask_idxs:
            if args.model_type in ["Task2Box", "Task2BoxSmall", "Task2BoxLinear"]:
                box1, box2 = box_tensors[idx1], box_tensors[idx2]
                overlap_pred = torch.exp(box_vol(box_int(box1, box2)) - box_vol(box1))
            elif args.model_type in ["linear", "mlp"]:
                in1 = torch.reshape(x_init[idx1], (1,-1))
                in2 = torch.reshape(x_init[idx2], (1,-1))
                concat_inp = torch.concat([in1,in2], dim=1)
                overlap_pred = model(concat_inp)
            label = transformed_overlaps[idx1, idx2]
            orig_label = orig_overlaps[idx1, idx2]
            
            preds.append(overlap_pred.item())
            labels.append(label)
            orig_pred = get_tanh_inv(overlap_pred.item(), k=args.tanh_k)
            orig_preds.append(orig_pred)
            orig_labels.append(orig_label)
        orig_preds, preds = np.array(orig_preds), np.array(preds)
        orig_labels, labels = np.array(orig_labels), np.array(labels)
        rho, p_value = stats.spearmanr(labels, preds)
        rho_orig, p_value_orig = stats.spearmanr(orig_labels, orig_preds)
        logger.debug(f"rho: {rho}, p_value: {p_value}. num samples: {len(labels)}")
        logger.debug(f"rho_orig: {rho_orig}, p_value_orig: {p_value_orig}")
    return rho, p_value, rho_orig, p_value_orig, orig_preds, orig_labels

def get_tanh(x, k=30):
    return (np.exp(k*x) - np.exp(-k*x))/(np.exp(k*x) + np.exp(-k*x))

def get_tanh_inv(x, k=30):
    return 0.5*np.log((1+(x/k))/(1-(x/k)))

if __name__=="__main__":
    args = parser.parse_args()
    logger.debug(f"args: {args}")
    if not os.path.exists(args.ckpts_dir):
        os.makedirs(args.ckpts_dir, exist_ok=True)

    tmp_args = vars(args)
    with open(f"{args.ckpts_dir}/{DATE_STR}_case{args.case_num:02d}_taskonomy_boxes{args.box_dim:02d}__params.json", "w") as f:
        json.dump(tmp_args, f, indent=4)
    
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Load input features for each dataset/task (e.g., for 100 tasks x should have shape of (100,in_dim))
    with open(f"../embeddings/{args.link_pred_src}/taskonomy_{args.case_num:02d}.pickle", "rb") as f:
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

    datasets_val = all_data["val_datasets"]
    datasets_test = all_data["test_datasets"]

    full_overlaps = all_data["full_overlap"]
    full_overlaps_transf = get_tanh(full_overlaps, k=args.tanh_k)

    datasets_train = all_data["datasets"]
    dists_key_name = "taskonomy_dists"
    train_overlaps = all_data["overlap"]
    train_overlaps_transf = get_tanh(train_overlaps, k=args.tanh_k)
    train_dists = all_data[dists_key_name]
    train_dists = np.minimum(train_dists, 10000)
    train_dists = train_dists/np.max(np.abs(train_dists))   # normalize distances

    data_key_name = "taskonomy_train_data"
    in_feats = all_data[data_key_name]
    in_dim = in_feats.shape[1]
    x_init = torch.from_numpy(in_feats).type(torch.FloatTensor)  # shape: (num_tasks, in_dim)    # fixed input features

    # For unseen/novel datasets
    if len(datasets_val)>0:
        val_in_feats = all_data["taskonomy_val_data"]
        x_val = torch.from_numpy(val_in_feats).type(torch.FloatTensor)  # shape: (num_tasks, in_dim)    # fixed input features
    test_in_feats = all_data["taskonomy_test_data"]
    x_test = torch.from_numpy(test_in_feats).type(torch.FloatTensor)  # shape: (num_tasks, in_dim)    # fixed input features

    if args.model_type == "Task2Box":
        model = BoxModel(box_dim=args.box_dim, in_dim=in_dim)
    elif args.model_type == "Task2BoxSmall":
        model = BoxModelSmall(box_dim=args.box_dim, in_dim=in_dim)
    elif args.model_type == "Task2BoxLinear":
        model = BoxModelLinear(box_dim=args.box_dim, in_dim=in_dim)
    elif args.model_type in ["linear", "mlp"]:
        train_dataset_load = TaskonomyPairData(in_feats=in_feats, mask_idxs=train_mask_idxs, train_overlaps_transf=train_overlaps_transf)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset_load,
            batch_size=args.batch_size,
            shuffle=True, num_workers=0)
        if args.model_type == "linear":
            model = LinearModel(in_dim=in_dim*2)
        elif args.model_type == "mlp":
            model = MLPModel(in_dim=in_dim*2)

    D_similarity = np.maximum(train_overlaps_transf, EPS)
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
    for epoch in tqdm(range(args.num_epochs)):
        idx_perm = torch.randperm(x_init.shape[0])  # shuffle train data
        x_train = x_init[idx_perm]

        model.train()
        if args.model_type in ["Task2Box", "Task2BoxSmall", "Task2BoxLinear"]:
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
            losses["overlap_loss"].append(overlap_loss.item())
            losses["dist_loss"].append(dist_loss.item())
            losses["vol_loss"].append(vol_loss.item())
            losses["box_reg"].append(box_reg.item())
        elif args.model_type in ["linear", "mlp"]:
            train_loss = train_linear(args, model, train_loader, optimizer)
        
        losses["train_loss"].append(train_loss.item())
        plot_losses(losses, epoch, DATE_STR, args)

        # Visualize boxes and save model checkpoint
        if epoch%(int(args.num_epochs*args.pct_saving_interval/100))==0:
            if args.model_type in ["Task2Box", "Task2BoxSmall", "Task2BoxLinear"]:
                logger.debug(f"train_loss: {train_loss}, overlap_loss: {overlap_loss}, dist_loss: {dist_loss}, vol_loss: {vol_loss}, box_reg: {box_reg}")
                display_boxes(out_boxes.detach().cpu().numpy(), list(range(len(x_init))), DATE_STR, args, epoch)
            else:
                logger.debug(f"train_loss: {train_loss}")
            
            logger.debug("Evaluating on Validation Set")
            val_rho, val_p_value, val_rho_orig, val_p_value_orig, val_orig_preds, val_orig_labels = evaluate(
                args, model, x_init, val_mask_idxs, box_vol, box_int, train_overlaps_transf, train_overlaps
            )
            logger.debug("Evaluating on Test Set")
            test_rho, test_p_value, test_rho_orig, test_p_value_orig, test_orig_preds, test_orig_labels = evaluate(
                args, model, x_init, test_mask_idxs, box_vol, box_int, train_overlaps_transf, train_overlaps
            )

            # Evaluate on new datasets
            if len(datasets_val)>0:
                logger.debug(f"Evaluating on NOVEL val datasets")
                val_new_rho, val_new_p_value, val_new_rho_orig, val_new_p_value_orig, val_new_orig_preds, val_new_orig_labels = evaluate_on_new_datasets(
                    args, model, x_train, x_val, datasets_train, datasets_val, box_vol, box_int, full_overlaps_transf, full_overlaps
                )
            logger.debug(f"Evaluating on NOVEL test datasets")
            test_new_rho, test_new_p_value, test_new_rho_orig, test_new_p_value_orig, test_new_orig_preds, test_new_orig_labels = evaluate_on_new_datasets(
                args, model, x_train, x_test, datasets_train, datasets_test, box_vol, box_int, full_overlaps_transf, full_overlaps
            )
            if args.model_type == "Task2Box":
                model_ckpt_fp = os.path.join(args.ckpts_dir, f"{DATE_STR}_case{args.case_num:02d}_taskonomy_{epoch:02d}_box{args.box_dim:02d}_model.pth")
            elif args.model_type in ["Task2BoxSmall", "Task2BoxLinear"]:
                model_ckpt_fp = os.path.join(args.ckpts_dir, f"{DATE_STR}_{args.model_type}_case{args.case_num:02d}_taskonomy_{epoch:02d}_box{args.box_dim:02d}_model.pth")
            else:
                model_ckpt_fp = os.path.join(args.ckpts_dir, f"{DATE_STR}_{args.model_type}_case{args.case_num:02d}_taskonomy_{epoch:02d}_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses': losses,

                "datasets": datasets_train,
                "datasets_val": datasets_val,
                "datasets_test": datasets_test,
                "test_metrics": [test_rho, test_p_value, test_rho_orig, test_p_value_orig],
                "val_metrics": [val_rho, val_p_value, val_rho_orig, val_p_value_orig],
                "val_orig_preds": val_orig_preds,
                "val_orig_labels": val_orig_labels,
                "test_orig_preds": test_orig_preds,
                "test_orig_labels": test_orig_labels,

                "novel_test_metrics": [test_new_rho, test_new_p_value, test_new_rho_orig, test_new_p_value_orig, test_new_orig_preds, test_new_orig_labels],
                }, model_ckpt_fp)
            
            
    # Save for final epoch
    if args.model_type in ["Task2Box", "Task2BoxSmall", "Task2BoxLinear"]:
        logger.debug(f"train_loss: {train_loss}, overlap_loss: {overlap_loss}, dist_loss: {dist_loss}, vol_loss: {vol_loss}, box_reg: {box_reg}")
        display_boxes(out_boxes.detach().cpu().numpy(), list(range(len(x_init))), DATE_STR, args, epoch)
    else:
        logger.debug(f"train_loss: {train_loss}")

    logger.debug("Evaluating on Validation Set")
    val_rho, val_p_value, val_rho_orig, val_p_value_orig, val_orig_preds, val_orig_labels = evaluate(
        args, model, x_init, val_mask_idxs, box_vol, box_int, train_overlaps_transf, train_overlaps
    )
    logger.debug("Evaluating on Test Set")
    test_rho, test_p_value, test_rho_orig, test_p_value_orig, test_orig_preds, test_orig_labels = evaluate(
        args, model, x_init, test_mask_idxs, box_vol, box_int, train_overlaps_transf, train_overlaps
    )

    # Evaluate on new datasets
    if len(datasets_val)>0:
        logger.debug(f"Evaluating on NOVEL val datasets")
        val_new_rho, val_new_p_value, val_new_rho_orig, val_new_p_value_orig, val_new_orig_preds, val_new_orig_labels = evaluate_on_new_datasets(
            args, model, x_train, x_val, datasets_train, datasets_val, box_vol, box_int, full_overlaps_transf, full_overlaps
        )
    logger.debug(f"Evaluating on NOVEL test datasets")
    test_new_rho, test_new_p_value, test_new_rho_orig, test_new_p_value_orig, test_new_orig_preds, test_new_orig_labels = evaluate_on_new_datasets(
        args, model, x_train, x_test, datasets_train, datasets_test, box_vol, box_int, full_overlaps_transf, full_overlaps
    )


    if args.model_type == "Task2Box":
        model_ckpt_fp = os.path.join(args.ckpts_dir, f"{DATE_STR}_case{args.case_num:02d}_taskonomy_{epoch:02d}_box{args.box_dim:02d}_model.pth")
    elif args.model_type in ["Task2BoxSmall", "Task2BoxLinear"]:
        model_ckpt_fp = os.path.join(args.ckpts_dir, f"{DATE_STR}_{args.model_type}_case{args.case_num:02d}_taskonomy_{epoch:02d}_box{args.box_dim:02d}_model.pth")
    else:
        model_ckpt_fp = os.path.join(args.ckpts_dir, f"{DATE_STR}_{args.model_type}_case{args.case_num:02d}_taskonomy_{epoch:02d}_model.pth")

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,


        "datasets": datasets_train,
        "datasets_val": datasets_val,
        "datasets_test": datasets_test,
        "test_metrics": [test_rho, test_p_value, test_rho_orig, test_p_value_orig],
        "val_metrics": [val_rho, val_p_value, val_rho_orig, val_p_value_orig],
        "val_orig_preds": val_orig_preds,
        "val_orig_labels": val_orig_labels,
        "test_orig_preds": test_orig_preds,
        "test_orig_labels": test_orig_labels,

        "novel_test_metrics": [test_new_rho, test_new_p_value, test_new_rho_orig, test_new_p_value_orig, test_new_orig_preds, test_new_orig_labels],
        }, model_ckpt_fp)