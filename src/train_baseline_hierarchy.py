import torch
import torch.nn as nn
import numpy as np

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

from utils.dataloader import PairData
from utils.model import MLPModel, LinearModel

EPS = 1e-12
DATE_STR = datetime.now().strftime("%Y%m%d-%H%M%S")
parser = argparse.ArgumentParser(description='MLP/Linear training script')

parser.add_argument('--model_type', default="linear", type=str,
                    help='Type of model to use for training. Choices: ["linear", "mlp"]')
parser.add_argument('--dataset', default="cubinat", type=str,
                    help='Dataset to use (cubinat, imagenet)')
parser.add_argument('--case_num', default=1, type=int,
                    help='Case number for training/evaluation [1-10] ')
parser.add_argument('--link_pred_src', default="link_pred_new50", type=str,
                    help='Where to get train data')
parser.add_argument('--gt_pairs_fp', default="../data/hierarchy_both.csv", type=str,
                    help='Directory for saving results')
parser.add_argument('--data_root', default="./data", type=str,
                    help='Directory for saving results')
parser.add_argument('--ckpts_dir', default="../ckpts_linear_link_pred", type=str,
                    help='Directory where dataset cache is downloaded/stored')
parser.add_argument('--batch_size', default=16, type=int,
                    help='batch size. ')
parser.add_argument('--feat_type', default="clip_gauss", type=str,
                    help='Model used for extracting embeddings ("clip_ave", "clip_gauss", "fim")')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--num_epochs', default=1500, type=int,
                    help='Number of epochs for training box embeddings')



parser.add_argument('--lr', default=0.001, type=float,
                    help='Learning rate for optimizer')
parser.add_argument('--optimizer', default="adam", type=str,
                    help='Type of optimizer to use ("adam", "sgd")')
parser.add_argument('--is_gradient_clip', default=0, type=int,
                    help='1 to do gradient clipping. 0 to NOT do gradient clipping')


parser.add_argument('--pct_saving_interval', default=5, type=int,
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
    plt.savefig(f"{args.ckpts_dir}/{date_str}_{args.model_type}_{args.feat_type}_case{args.case_num:02d}_{plot_type}.jpg", bbox_inches='tight')
    plt.close()


def train(args, model, train_loader, optimizer,):
    loss_fn = nn.BCELoss()
    
    for x,y in train_loader:
        if x.shape[0]==1:   # skip batches with just 1 member
            continue
        y = y.type(torch.FloatTensor)
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

def linear_evaluate(args, model, val_loader, thresh=None):
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
            return optimal_threshold
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

if __name__=="__main__":
    args = parser.parse_args()
    logger.debug(f"args: {args}")
    if not os.path.exists(args.ckpts_dir):
        os.makedirs(args.ckpts_dir, exist_ok=True)

    tmp_args = vars(args)
    with open(f"{args.ckpts_dir}/{DATE_STR}_{args.model_type}_{args.feat_type}_case{args.case_num:02d}__params.json", "w") as f:
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

    data_key_name = f"{args.feat_type}_train_data"
    in_feats = all_data[data_key_name]
    in_dim = in_feats.shape[1]
    x_init = torch.from_numpy(in_feats).type(torch.FloatTensor)  # shape: (num_tasks, in_dim)    # fixed input features


    # The following is for evaluation
    gt_data_pairs = pd.read_csv(args.gt_pairs_fp, index_col=0)
    gt_data_pairs.dropna(inplace=True)
    logger.debug(f"Loaded GT pairs shape: {gt_data_pairs.shape}.")

    train_labels = get_label_from_pair_idxs(train_mask_idxs, gt_data_pairs, datasets_train)
    val_labels = get_label_from_pair_idxs(val_mask_idxs, gt_data_pairs, datasets_train)
    test_labels = get_label_from_pair_idxs(test_mask_idxs, gt_data_pairs, datasets_train)

    if args.model_type == "linear":
        model = LinearModel(in_dim=in_dim*2)
    elif args.model_type == "mlp":
        model = MLPModel(in_dim=in_dim*2)
    else:
        raise NotImplementedError

    train_dataset = PairData(in_feats, train_mask_idxs, train_labels)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True, num_workers=0)
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
    
    if args.optimizer == "sgd":
        optimizer =  torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == "adam":
        optimizer =  torch.optim.Adam(model.parameters(), lr=args.lr)

    
    losses = {"train_loss": []}
    metrics = { "epoch": [], "val_f1": [],"train_f1": [], "val_prec": [],"train_prec": [], "val_rec": [], "train_rec": []}
    for epoch in tqdm(range(args.num_epochs)):
        idx_perm = torch.randperm(x_init.shape[0])  # shuffle train data
        x_train = x_init[idx_perm]

        model.train()
        train_loss = train(
            args,
            model,
            train_loader,
            optimizer,
        )
        
        losses["train_loss"].append(train_loss.item())
        plot_losses(losses, epoch, DATE_STR, args)

        # Visualize boxes and save model checkpoint
        if epoch%(int(args.num_epochs*args.pct_saving_interval/100))==0:
            logger.debug(f"train_loss: {train_loss}")
            
            logger.debug("Validation set evaluation")
            val_f1, val_prec, val_rec = linear_evaluate(args, model, val_loader, thresh=0.5)
            metrics["epoch"].append(epoch)
            metrics["val_f1"].append(val_f1)
            metrics["val_prec"].append(val_prec)
            metrics["val_rec"].append(val_rec)

            model_ckpt_fp = os.path.join(args.ckpts_dir, f"{DATE_STR}_{args.model_type}_{args.feat_type}_case{args.case_num:02d}_{epoch:02d}_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses': losses,
                'metrics': metrics,
                }, model_ckpt_fp)
            plot_losses(metrics, epoch, DATE_STR, args, plot_type="metrics")
            
    # Save for final epoch
    logger.debug(f"train_loss: {train_loss}")
    
    logger.debug("Validation set evaluation")
    val_f1, val_prec, val_rec = linear_evaluate(args, model, val_loader, thresh=0.5)
    metrics["epoch"].append(epoch)
    metrics["val_f1"].append(val_f1)
    metrics["val_prec"].append(val_prec)
    metrics["val_rec"].append(val_rec)

    logger.debug("Train set evaluation")
    train_f1, train_prec, train_rec = linear_evaluate(args, model, train_loader, thresh=0.5)
    metrics["train_f1"].append(train_f1)
    metrics["train_prec"].append(train_prec)
    metrics["train_rec"].append(train_rec)
    model_ckpt_fp = os.path.join(args.ckpts_dir, f"{DATE_STR}_{args.model_type}_{args.feat_type}_case{args.case_num:02d}_{epoch:02d}_model.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'metrics': metrics,
        }, model_ckpt_fp)
    plot_losses(metrics, epoch, DATE_STR, args, plot_type="metrics")
    
    optim_thresh = linear_evaluate(args, model, val_loader, thresh=None)
    f1, prec, rec = linear_evaluate(args, model, test_loader, thresh=optim_thresh)