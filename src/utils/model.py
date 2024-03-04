import torch.nn as nn
from loguru import logger


class BoxModel(nn.Module):
    def __init__(self, in_dim=2048, box_dim=2) -> None:
        super(BoxModel, self).__init__()
        self.box_dim = box_dim
        self.in_dim = in_dim

        self.backbone = nn.Sequential(
            nn.Linear(in_dim,1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),

            nn.Linear(1024,256),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256,128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )
        self.head1 = nn.Linear(128, box_dim)    # lower left coor of box
        self.head2 = nn.Sequential(             # size of the box to get upper right coor of box
            nn.Linear(128, box_dim),
            nn.Sigmoid(),   # need positive sizes
        )

    def forward(self, x):
        feats = self.backbone(x)
        ll_coor = self.head1(feats)
        sizes = self.head2(feats)
        return ll_coor, sizes

class MLPModel(nn.Module):
    def __init__(self, in_dim=2048) -> None:
        super(MLPModel, self).__init__()
        self.in_dim = in_dim

        self.backbone = nn.Sequential(
            nn.Linear(in_dim,1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),

            nn.Linear(1024,256),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256,128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            
            nn.Linear(128,1),
            nn.Sigmoid(),   # 0 to 1
        )

    def forward(self, x):   # x is the concatenated [embedding1,embedding2]
        return self.backbone(x)

class LinearModel(nn.Module):
    def __init__(self, in_dim=2048) -> None:
        super(LinearModel, self).__init__()
        self.in_dim = in_dim

        self.backbone = nn.Sequential(
            nn.Linear(in_dim,1),
            nn.Sigmoid(),   
        )

    def forward(self, x):   # x is the concatenated [embedding1,embedding2]
        return self.backbone(x)
    
class BoxModelSmall(nn.Module):
    def __init__(self, in_dim=2048, box_dim=2) -> None:
        super(BoxModelSmall, self).__init__()
        self.box_dim = box_dim
        self.in_dim = in_dim

        self.backbone = nn.Sequential(
            nn.Linear(in_dim,16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
        )
        self.head1 = nn.Linear(16, box_dim)    # lower left coor of box
        self.head2 = nn.Sequential(             # size of the box to get upper right coor of box
            nn.Linear(16, box_dim),
            nn.Sigmoid(),   # need positive sizes
        )

    def forward(self, x):
        feats = self.backbone(x)
        ll_coor = self.head1(feats)
        sizes = self.head2(feats)
        return ll_coor, sizes
    
class BoxModelLinear(nn.Module):
    def __init__(self, in_dim=2048, box_dim=2) -> None:
        super(BoxModelLinear, self).__init__()
        self.box_dim = box_dim
        self.in_dim = in_dim

        self.head1 = nn.Linear(in_dim, box_dim)    # lower left coor of box
        self.head2 = nn.Sequential(             # size of the box to get upper right coor of box
            nn.Linear(in_dim, box_dim),
            nn.Sigmoid(),   # need positive sizes
        )

    def forward(self, x):
        ll_coor = self.head1(x)
        sizes = self.head2(x)
        return ll_coor, sizes