from typing import Optional

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor
from layers import HalfNLHconv
from HGNN import hgnn
from sampling import negative_sampling

class gen(torch.nn.Module):
    def __init__(self,args,dim):
        super(gen, self).__init__()
        self.aggr=args.aggregate
        self.dropout=args.dropout

        #
        self.V2EConvs = nn.ModuleList()
        #
        self.E2VConvs = nn.ModuleList()

        self.V2EConvs.append(HalfNLHconv(in_dim=dim,  # dim of the graph
                                         hid_dim=args.MLP_hidden,
                                         out_dim=args.MLP_hidden,
                                         num_layers=args.MLP_num_layers,
                                         dropout=args.dropout,
                                         Normalization=args.normalization,
                                         InputNorm=True,
                                         heads=args.heads,
                                         attention=args.PMA))
        self.E2VConvs.append(HalfNLHconv(in_dim=args.MLP_hidden,
                                         hid_dim=args.MLP_hidden,
                                         out_dim=args.MLP_hidden,
                                         num_layers=args.MLP_num_layers,
                                         dropout=args.dropout,
                                         Normalization=args.normalization,
                                         InputNorm=True,
                                         heads=args.heads,
                                         attention=args.PMA))
        for _ in range(args.All_num_layers - 1):
            self.V2EConvs.append(HalfNLHconv(in_dim=args.MLP_hidden,
                                             hid_dim=args.MLP_hidden,
                                             out_dim=args.MLP_hidden,
                                             num_layers=args.MLP_num_layers,
                                             dropout=self.dropout,
                                             Normalization=self.NormLayer,
                                             InputNorm=self.InputNorm,
                                             heads=args.heads,
                                             attention=args.PMA))
            self.E2VConvs.append(HalfNLHconv(in_dim=args.MLP_hidden,
                                             hid_dim=args.MLP_hidden,
                                             out_dim=args.MLP_hidden,
                                             num_layers=args.MLP_num_layers,
                                             dropout=self.dropout,
                                             Normalization=self.NormLayer,
                                             InputNorm=self.InputNorm,
                                             heads=args.heads,
                                             attention=args.PMA))
        # self.hgnn=hgnn(args,dim)
        hidden = args.MLP_hidden
        self.encoder_mean_node = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True),nn.Linear(hidden, hidden))
        self.encoder_std_node = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True),nn.Linear(hidden, hidden), nn.Softplus())
        self.encoder_mean_he = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True),nn.Linear(hidden, hidden))
        self.encoder_std_he = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden),nn.Softplus())

        self.decoder = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))
        self.sigmoid = nn.Sigmoid()
        self.bceloss = nn.BCELoss(reduction='mean')
        self.celoss = nn.CrossEntropyLoss()

    def forward(self,x,hyperedge_index,norm):

        #
        reversed_edge_index = torch.stack([hyperedge_index[1], hyperedge_index[0]], dim=0)
        #
        for i, _ in enumerate(self.V2EConvs):
            # print(edge_index.dtype, edge_index[0].min(), edge_index[0].max(), edge_index[1].min(), edge_index[1].max())
            x_he = F.relu(self.V2EConvs[i](x, hyperedge_index, norm, aggr=self.aggr))
            #                 x = self.bnV2Es[i](x)
            x_x = F.dropout(x_he, p=self.dropout, training=self.training)
            x_node = F.relu(self.E2VConvs[i](x_x, reversed_edge_index, norm, self.aggr))
            #                 x = self.bnE2Vs[i](x)
            x = F.dropout(x_node, p=self.dropout, training=self.training)

        x_mean_node = self.encoder_mean_node(x_node)
        x_std_node = self.encoder_std_node(x_node)
        gaussian_noise = torch.randn(x_mean_node.shape).cuda()
        x_node = gaussian_noise * x_std_node + x_mean_node

        x_mean_he = self.encoder_mean_he(x_he)
        x_std_he = self.encoder_std_he(x_he)
        gaussian_noise = torch.randn(x_mean_he.shape).cuda()
        x_he = gaussian_noise * x_std_he + x_mean_he

        edge_pred = self.decoder(x_node[hyperedge_index[0]] * x_he[hyperedge_index[1]]).squeeze()
        loss_edge = self.celoss(edge_pred, torch.ones(edge_pred.shape[0], dtype=torch.long).to(edge_pred.device))
        # negative
        negative_hyperedge_index = negative_sampling(hyperedge_index, num_nodes=(hyperedge_index[0].max() + 1, hyperedge_index[1].max() + 1))
        edge_pred_neg = self.decoder(x_node[negative_hyperedge_index[0]] * x_he[negative_hyperedge_index[1]]).squeeze()
        loss_edge += self.celoss(edge_pred_neg, torch.zeros(edge_pred_neg.shape[0], dtype=torch.long).to(edge_pred.device))
        #
        edge_pred = F.gumbel_softmax(edge_pred,tau=0.1, hard=True)
        # edge_pred = F.softmax(edge_pred,dim=1)



        # edge_pred = F.gumbel_softmax(edge_pred,hard=True)
        edge_pred = edge_pred[:, 1]  # (204,)

        # loss_edge = self.celoss(edge_pred, torch.ones(edge_pred.shape[0], dtype=torch.long).to(edge_pred.device))

        kl_divergence_node = - 0.5 * (1 + 2 * torch.log(x_std_node) - x_mean_node ** 2 - x_std_node ** 2).sum(
            dim=1).mean()
        kl_divergence_node = kl_divergence_node / x_node.size(0)

        kl_divergence_he = - 0.5 * (1 + 2 * torch.log(x_std_he) - x_mean_he ** 2 - x_std_he ** 2).sum(dim=1).mean()
        kl_divergence_he = kl_divergence_he / x_he.size(0)

        loss = (loss_edge + kl_divergence_node + kl_divergence_he).mean()

        return edge_pred,loss

    def reset_parameters(self):

        for layer in self.V2EConvs:
            layer.reset_parameters()
        for layer in self.E2VConvs:
            layer.reset_parameters()




