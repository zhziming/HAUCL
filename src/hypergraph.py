from itertools import permutations
import torch
from utils import contrastive_loss
import torch.nn as nn
from generator import gen
from HGNN import hgnn
import numpy as np

class HyperGCN(nn.Module):
    #
    def __init__(self,args,dropout=0.5,n_dim=512,lamda=0.5,alpha=0.1,n_speakers=9):
        super(HyperGCN, self).__init__()
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.n_dim=n_dim
        self.args = args
        self.fc=nn.Linear(n_dim,args.graph_dim)
        # ------------------------------------
        if args.use_sp:
            self.speaker_cat_methd = args.speaker_fusion_methd
            self.speaker_embeddings = nn.Embedding(n_speakers, args.dim)
        if args.use_position:
            self.l_pos = PositionalEncoding(args.dim)
            self.a_pos = PositionalEncoding(args.dim)
            self.v_pos = PositionalEncoding(args.dim)
        #------------------------------------

        # hyconv
        self.hg =hgnn(args,args.graph_dim)
        self.hyperedge_weight = nn.Parameter(torch.ones(1000))
        self.EW_weight = nn.Parameter(torch.ones(5200))
        self.hyperedge_attr1 = nn.Parameter(torch.rand(n_dim))
        self.hyperedge_attr2 = nn.Parameter(torch.rand(n_dim))
        # args.PMA = False
        # args.aggregate = 'mean'
        # self.hg = SetGNN(args,n_dim)
        # ------------------------------------

        # gen
        self.generator = gen(args,n_dim)

        # cl
        self.linear1=nn.Linear(n_dim, n_dim)
        self.relu=nn.ReLU()
        self.linear2=nn.Linear(n_dim, n_dim)
        # ------------------------------------

        # gcn
        # ------------------------------------
        if args.use_gcn:
            if args.Dataset == 'MELD':
                nclass = 7
            else:
                nclass = 6

            from model_GCN import GCNII_lyc
            self.graph_net = GCNII_lyc(nfeat=n_dim, nlayers=args.num_layers, nhidden=n_dim, nclass=nclass,
                               dropout=dropout, lamda=lamda, alpha=alpha, variant=True,
                               return_feature=True, use_residue=args.use_residue)

    #（354，512）（list:32）
    def forward(self,a, v, l, dia_len, qmask,train,e,i):

        # position
        if self.args.use_position:
            l = self.l_pos(l, dia_len)
            a = self.a_pos(a, dia_len)
            v = self.v_pos(v, dia_len)

        # speaker
        if self.args.use_sp :
            qmask = torch.cat([qmask[:x, i, :] for i, x in enumerate(dia_len)], dim=0)  # (179,9) speaker-onehot
            spk_idx = torch.argmax(qmask, dim=-1)  #（354）--speaker
            spk_emb_vector = self.speaker_embeddings(spk_idx) # (354,512)
            # speaker
            if self.speaker_cat_methd == 'fuse':
                l += spk_emb_vector
                # a += spk_emb_vector
                # v += spk_emb_vector
            else:
                l = torch.cat([l,spk_emb_vector],dim=-1)
                a = torch.cat([a, spk_emb_vector], dim=-1)
                v = torch.cat([v, spk_emb_vector], dim=-1)

        #
        hyperedge_index, edge_index, feature, batch, hyperedge_type = self.create_hyper_index(a, v, l, dia_len) # （354，1024）list(32)

        #
        # hyperedge_weight = self.hyperedge_weight[0:hyperedge_index[1].max().item() + 1]  # 450个1
        # hyperedge_attr = self.hyperedge_attr1 * hyperedge_type + self.hyperedge_attr2 * (1 - hyperedge_type)  # 超边特征为（450，1024）
        EW_weight = self.EW_weight[0:hyperedge_index.size(1)]

        features = self.fc(feature)


        if self.args.use_cl  & self.args.use_g:
            # x,hyperedge_index,hyperedge_type
            pred1, loss_g1 = self.generator(features, hyperedge_index, EW_weight)
            # x,edge_index,norm,aug_weight=None

            out1,_= self.hg(features, hyperedge_index, hyperedge_type, pred1)
            # out1, _ = self.hg(out1, hyperedge_index, hyperedge_type, pred1)
            # out1, _ = self.hg(out1, hyperedge_index, hyperedge_type, pred1)

            pred2, loss_g2 = self.generator(features, hyperedge_index, EW_weight)

            out2,_ = self.hg(features, hyperedge_index, hyperedge_type, pred2)
            # out2, _ = self.hg(out2, hyperedge_index, hyperedge_type, pred2)
            # out2, _ = self.hg(out2, hyperedge_index, hyperedge_type, pred2)

            # loss_cl = loss(out_2, out_1)
            out1_aug = self.linear2(self.relu(self.linear1(out1)))
            out2_aug = self.linear2(self.relu(self.linear1(out1)))
            loss_cl=contrastive_loss(out2_aug,out1_aug)

            # loss_cl = ours_loss(out2, out1)
            # print(f"loss_cl:{loss_cl}")
            out=out1+out2
        elif self.args.use_g:
            pred, _ = self.generator(features, hyperedge_index, EW_weight)
            # features, hyperedge_index, hyperedge_type, bi_weight = None)
            out,_= self.hg(features, hyperedge_index, hyperedge_type,pred)
            loss_cl=0
            loss_g1 ,loss_g2=0,0
        elif self.args.use_cl:
            out1, _ = self.hg(features, hyperedge_index, hyperedge_type)
            out2, _ = self.hg(features, hyperedge_index, hyperedge_type)
            out=out1+out2
            loss_cl=contrastive_loss(out2,out1)
            loss_g1, loss_g2 = 0, 0
        else:
            out, _ = self.hg(features, hyperedge_index, hyperedge_type)

            loss_cl = 0
            loss_g1, loss_g2 = 0, 0
        if self.args.use_residue:
            #
            out = torch.cat([feature, out], dim=1) #
        out = self.reverse_features(dia_len, out)  # 3*dim

        if self.args.use_gcn:
            # MMGCN
            adj = self.create_big_adj(a, v, l, dia_len)
            features_gcn = torch.cat([a, v, l], dim=0).cuda()
            features_gcn = self.graph_net(features_gcn, None, qmask, adj)
            all_length = l.shape[0] if len(l) != 0 else a.shape[0] if len(a) != 0 else v.shape[0]
            features_gcn = torch.cat(
                [features_gcn[:all_length], features_gcn[all_length:all_length * 2], features_gcn[all_length * 2:all_length * 3]],
                dim=-1)
            #
            out = torch.cat([features_gcn, out], dim=1)


        return out ,loss_cl,(loss_g1+loss_g2)/2

    def create_hyper_index(self, a, v, l, dia_len):
        node_count = 0
        edge_count = 0
        batch_count = 0
        index1 = []
        index2 = []
        tmp = []
        batch = []
        hyperedge_type1 = []
        for i in dia_len:
            nodes = list(range(i * 3))
            nodes = [j + node_count for j in nodes]
            nodes_l = nodes[0:i * 3 // 3]
            nodes_a = nodes[i * 3 // 3:i * 3 * 2 // 3]
            nodes_v = nodes[i * 3 * 2 // 3:]
            index1 = index1 + nodes_l + nodes_a + nodes_v #
            for _ in range(i):
                index1 = index1 + [nodes_l[_]] + [nodes_a[_]] + [nodes_v[_]] #
            for _ in range(i + 3):
                if _ < 3:
                    index2 = index2 + [edge_count] * i
                else:
                    index2 = index2 + [edge_count] * 3
                edge_count = edge_count + 1
            # features
            if node_count == 0:
                ll = l[0:0 + i]
                aa = a[0:0 + i]
                vv = v[0:0 + i]
                features = torch.cat([ll, aa, vv], dim=0)
                temp = 0 + i
            else:
                ll = l[temp:temp + i]
                aa = a[temp:temp + i]
                vv = v[temp:temp + i]
                features_temp = torch.cat([ll, aa, vv], dim=0)
                features = torch.cat([features, features_temp], dim=0)
                temp = temp + i

            Gnodes = []
            Gnodes.append(nodes_l)
            Gnodes.append(nodes_a)
            Gnodes.append(nodes_v)
            for _ in range(i):
                Gnodes.append([nodes_l[_]] + [nodes_a[_]] + [nodes_v[_]])
            for ii, _ in enumerate(Gnodes):
                perm = list(permutations(_, 2))
                tmp = tmp + perm
            batch = batch + [batch_count] * i * 3
            batch_count = batch_count + 1
            hyperedge_type1 = hyperedge_type1 + [1] * i + [0] * 3
            node_count = node_count + i * 3

        index1 = torch.LongTensor(index1).view(1, -1)
        index2 = torch.LongTensor(index2).view(1, -1)
        hyperedge_index = torch.cat([index1, index2], dim=0).cuda()

        edge_index = torch.LongTensor(tmp).T.cuda()
        batch = torch.LongTensor(batch).cuda()
        hyperedge_type1 = torch.LongTensor(hyperedge_type1).view(-1, 1).cuda()

        return hyperedge_index, edge_index, features, batch, hyperedge_type1
    def reverse_features(self, dia_len, features):
        l = []
        a = []
        v = []
        for i in dia_len:
            ll = features[0:1 * i]
            aa = features[1 * i:2 * i]
            vv = features[2 * i:3 * i]
            features = features[3 * i:]
            l.append(ll)
            a.append(aa)
            v.append(vv)
        tmpl = torch.cat(l, dim=0)
        tmpa = torch.cat(a, dim=0)
        tmpv = torch.cat(v, dim=0)
        features = torch.cat([tmpl, tmpa, tmpv], dim=-1)
        return features