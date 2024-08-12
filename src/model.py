import torch.nn as nn
import torch.nn.functional as F
from hypergraph import HyperGCN
from utils import simple_batch_graphify
class Model(nn.Module):
    def __init__(self,args,D_a=300,D_v=342,D_t=1024,n_speakers=9,n_classes=7):
        super(Model, self).__init__()
        D_g=args.dim
        #
        self.normBNa = nn.BatchNorm1d(1024, affine=True)
        self.normBNb = nn.BatchNorm1d(1024, affine=True)
        self.normBNc = nn.BatchNorm1d(1024, affine=True)
        self.normBNd = nn.BatchNorm1d(1024, affine=True)
        #
        self.linear_a = nn.Linear(D_a, D_g)
        self.linear_v = nn.Linear(D_v, D_g)
        self.linear_t = nn.Linear(D_t, D_g)
        self.gru_t = nn.GRU(input_size=D_g, hidden_size=D_g // 2, num_layers=2, bidirectional=True,dropout=0.5)
        #
        self.dropout_ = nn.Dropout(args.dropout)

        #
        if args.use_sp & (args.speaker_fusion_methd=='gated') & args.use_position:
            graph_dim = 3 * D_g
        elif args.use_sp & (args.speaker_fusion_methd=='gated'):
            graph_dim = 2 * D_g
        else:
            graph_dim = D_g

        self.graph_model = HyperGCN(args, n_speakers=n_speakers,n_dim=graph_dim)

        # mmgcn
        if args.use_gcn:
            dim = args.graph_dim*3*3
        else:
            dim = args.graph_dim*2*3

        if not args.use_residue:
            dim = dim // 2

        self.smax_fc1 = nn.Linear(dim, 12*n_classes)
        self.smax_fc2 = nn.Linear(12*n_classes, n_classes)

    def forward(self,U_t,U_a,U_v,lengths,qmast,e,i,train=False):
        # =============roberta features
        [r1, r2, r3, r4] = U_t
        seq_len, _, feature_dim = r1.size()
        r1 = self.normBNa(r1.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        r2 = self.normBNb(r2.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        r3 = self.normBNc(r3.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        r4 = self.normBNd(r4.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        U_t = (r1 + r2 + r3 + r4) / 4  # (21,32,512)
        U_a = self.linear_a(U_a)   # (21,32,512)
        U_v = self.linear_v(U_v)   # (21,32,512)
        U_t=self.linear_t(U_t)  # (21,32,512)
        U_t, hidden_l = self.gru_t(U_t)  # (21,32,512)

        features_a, edge_index, edge_norm, edge_type, edge_index_lengths = simple_batch_graphify(U_a,lengths)  # (354,512)
        features_v, edge_index, edge_norm, edge_type, edge_index_lengths = simple_batch_graphify(U_v,lengths)  # (354,512)
        features_t, edge_index, edge_norm, edge_type, edge_index_lengths = simple_batch_graphify(U_t,lengths)  # (354,512)

        emotions_feat,loss_cl,loss_g= self.graph_model(features_a, features_v, features_t,lengths,qmast,train,e,i)
        # print(f"loss_cl:{loss_cl},loss_g:{loss_g}")
        emotions_feat = self.dropout_(emotions_feat)
        emotions_feat = nn.ReLU()(emotions_feat) #158,7680
        # log_prob = F.log_softmax(self.smax_fc2(self.smax_fc1(emotions_feat)), 1)
        log_prob = self.smax_fc2(self.smax_fc1(emotions_feat))

        return log_prob, edge_index, edge_norm, edge_type, edge_index_lengths,loss_cl,loss_g,emotions_feat