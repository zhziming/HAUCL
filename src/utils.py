import torch,numpy,random
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree
import torch
import numpy as np
import matplotlib.pyplot as plt
import math

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.5, alpha=1, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001

    def forward(self, logits, labels):
        """
        cal culates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
        """
        if labels.dim() > 2:
            labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
            labels = labels.transpose(1, 2)
            labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
        if logits.dim() > 3:
            logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
            logits = logits.transpose(2, 3)
            logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
        labels_length = logits.size(1)
        seq_length = logits.size(0)

        new_label = labels.unsqueeze(1)
        label_onehot = torch.zeros([seq_length, labels_length]).cuda().scatter_(1, new_label, 1)

        log_p = F.log_softmax(logits, -1)
        pt = label_onehot * log_p
        sub_pt = 1 - pt
        fl = -self.alpha * (sub_pt) ** self.gamma * log_p
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()
            
def seed_everything(seed=67137):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def simple_batch_graphify(features, lengths):
    edge_index, edge_norm, edge_type, node_features = [], [], [], []
    batch_size = features.size(1)
    for j in range(batch_size):
        node_features.append(features[:lengths[j], j, :])
    node_features = torch.cat(node_features, dim=0)
    node_features = node_features.cuda()
    return node_features, None, None, None, None

def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())

def contrastive_loss(x1, x2):
    T = 0.5
    # if args.dname in ["yelp", "coauthor_dblp", "walmart-trips-100"]:
    #     batch_size=1024
    # else:
    #     batch_size = None
    batch_size = None
    if batch_size is None:
        l1 = semi_loss(x1, x2, T)
        l2 = semi_loss(x2, x1, T)
    else:
        l1 = batched_semi_loss(x1, x2, batch_size, T)
        l2 = batched_semi_loss(x2, x1, batch_size, T)
    ret = (l1 + l2) * 0.5
    ret = ret.mean()
    return ret
    
def semi_loss(z1: torch.Tensor, z2: torch.Tensor, T):
    f = lambda x: torch.exp(x / T)
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))
    return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))