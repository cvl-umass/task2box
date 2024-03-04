from torch.utils.data.dataset import Dataset

import os
import torch
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
from loguru import logger


class PairData(Dataset):
    def __init__(self, in_feats, mask_idxs, labels):
        # Create features from mask_idxs
        self.labels = labels

        all_feats = []
        for idx1, idx2 in mask_idxs:
            x1 = in_feats[idx1]
            x2 = in_feats[idx2]
            inp = np.concatenate((x1,x2),axis=0)
            all_feats.append(np.reshape(inp, (-1,)))

        self.all_feats = all_feats

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.all_feats[index], self.labels[index]
    
class EmbedPairData(Dataset):
    def __init__(self, train_feats, test_feats, train_test_idxs, labels):
        # Create features from mask_idxs
        self.labels = labels

        all_feats = []
        for idx1, idx2 in train_test_idxs:
            x1 = train_feats[idx1]
            x2 = test_feats[idx2]
            inp = np.concatenate((x1,x2),axis=0)
            all_feats.append(np.reshape(inp, (-1,)))

        self.all_feats = all_feats

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.all_feats[index], self.labels[index]
    
class TaskonomyPairData(Dataset):
    def __init__(self, in_feats, mask_idxs, train_overlaps_transf):
        # Create features from mask_idxs
        
        labels = []
        all_feats = []
        for idx1, idx2 in mask_idxs:
            x1 = in_feats[idx1]
            x2 = in_feats[idx2]
            inp = np.concatenate((x1,x2),axis=0)
            all_feats.append(np.reshape(inp, (-1,)))

            label = train_overlaps_transf[idx1, idx2]
            labels.append(label)

        self.all_feats = all_feats
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.all_feats[index], self.labels[index]