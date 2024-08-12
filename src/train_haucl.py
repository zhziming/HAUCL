from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import torch
from argparse import Namespace
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
import torch.optim as optim
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import argparse

from dataloader import get_MELD_loaders
from utils import seed_everything, FocalLoss
from model import Model

def train_or_eval_graph_model(model, loss_function, dataloader,epoch,optimizer=None,train=False):
    losses, preds, labels = [], [], []
    if train:
        model.train()
    else:
        model.eval()
    seed_everything()
    i=0
    tensor_list = []
    label_list= []
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        #
        textf1, textf2, textf3, textf4, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]]
        lengths = [(umask[j] == 1).nonzero(as_tuple=False).tolist()[-1][0] + 1 for j in range(len(umask))]
        #
        log_prob, e_i, e_n, e_t, e_l,loss_cl,loss_g,tensor = model([textf1, textf2, textf3, textf4], acouf, visuf,lengths, qmask,epoch,i,train)
        i=i+1
        label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
        label1=label
        loss = loss_function(log_prob, label)

        #
        if args.use_cl:
            loss += args.cl * loss_cl

        if args.use_g:
            loss += args.g * loss_g

        preds.append(torch.argmax(log_prob, 1).cpu().numpy())
        labels.append(label.cpu().numpy())
        losses.append(loss.item())

        if train:
            loss.backward() 
            optimizer.step()
        else:
            #
            tensor_list.append(tensor.cpu().detach().numpy())
            label_list.append(label1.cpu().detach().numpy())


    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], []
    labels = np.array(labels)
    preds = np.array(preds)
    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 2)

    return avg_loss, avg_accuracy, labels, preds, avg_fscore
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Dataset', default='MELD', help='dataset to train, test or valid, MELD/IEMOCAP')
    parser.add_argument('--graph_dim', type=int, default=512, metavar='E', help='graph embedding dimension')
    parser.add_argument('--dim', type=int, default=512, metavar='E', help='multimodal embedding dimension')
    parser.add_argument('--use_g', action='store_true', default=True,help='whether to use vhgae as generative model')
    parser.add_argument('--use_cl', action='store_true', default=True, help='whether to use contrastive loss as multi-task training')
    parser.add_argument('--g', type=float, default=0.1, metavar='E', help='weight of generative loss')
    parser.add_argument('--cl', type=float, default=0.5, metavar='E', help='weight of contrastive loss')
    parser.add_argument('--use_gcn', action='store_true', default=False, help='whether to use mmgcn')
    parser.add_argument('--use_position', action='store_true', default=False, help='whether to use position encoding')
    parser.add_argument('--batch-size', type=int, default=12 , metavar='BS', help='batch size')
    parser.add_argument('--use_sp', action='store_true', default=True, help='whether to train speaker embedding')
    parser.add_argument('--speaker_fusion_methd', default='fuse',help='method to use speaker embedding: fuse/gated')
    parser.add_argument('--MLP_hidden', default=512, type=int, help='MLP layer hidden dimension') 
    parser.add_argument('--MLP_num_layers', default=1, type=int, help='Number of MLP layer')
    parser.add_argument('--dropout', type=float, default=0.4, metavar='E', help='dropout')
    parser.add_argument('--normalization', default='ln')
    parser.add_argument('--PMA', action='store_true', default=True, help='whether using attention mechanism for message passing')
    parser.add_argument('--heads', default=1, type=int)
    parser.add_argument('--aggregate', default='mean', choices=['sum', 'mean'])
    parser.add_argument('--All_num_layers', default=1, type=int, help='Number of hypergraph convolution layer')
    parser.add_argument('--use_residue', action='store_true', default=True,help='whether to use residue connection from raw feature')
    parser.add_argument('--lr', type=float, default=0.00005, metavar='LR', help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00003, metavar='weight_decay', help='weight decay')
    parser.add_argument('--epochs', type=int, default=10 , metavar='E', help='number of epochs')

    args = parser.parse_args()
    seed_everything()
    print(f"The configuration of this experiment is:{args}")

    name = args.Dataset + '_' + str(args.graph_dim)
    if args.use_g:name+='_use_g'
    if args.use_cl:name+='_use_cl'
    if args.use_gcn: name += '_use_gcn'
    if args.use_position: name += '_use_position'
    writer = SummaryWriter(name)

    if args.Dataset == 'MELD':
        model=Model(args).cuda()
        train_loader, valid_loader, test_loader = get_MELD_loaders(valid=0,num_workers=2,batch_size=args.batch_size)

    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.Dataset == 'MELD':
        loss_function = FocalLoss()

    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []
    
    # train/valid/test
    for e in range(args.epochs):
        start_time = time.time()
        train_loss, train_acc, _, _, train_fscore = train_or_eval_graph_model(model,loss_function,train_loader,e,optimizer,True)
        # print(f"train,loss:{train_loss},acc:{train_acc},fscore;{train_fscore}")
        # valid_loss, valid_acc, _, _, valid_fscore = train_or_eval_graph_model(model,loss_function,valid_loader,e)
        # print(f"valid,loss:{valid_loss},acc:{valid_acc},fscore;{valid_fscore}")
        test_loss, test_acc, test_label, test_pred, test_fscore = train_or_eval_graph_model(model,loss_function,test_loader,e)
        # print(f"test,loss:{test_loss},acc:{test_acc},fscore;{test_fscore}")
        all_fscore.append(test_fscore)
    
        if best_loss == None or best_loss > test_loss:
            best_loss, best_label, best_pred = test_loss, test_label, test_pred
        if best_fscore == None or best_fscore < test_fscore:
            best_fscore = test_fscore
            best_label, best_pred = test_label, test_pred
        writer.add_scalar('test: loss', test_loss, e)
        writer.add_scalar('test: accuracy', test_acc, e)
        writer.add_scalar('test: fscore', test_fscore, e)
        writer.add_scalar('train: loss', train_loss, e)
        writer.add_scalar('train: accuracy', train_acc, e)
        writer.add_scalar('train: fscore', train_fscore, e)
        print(
            'epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'. \
            format(e + 1, train_loss, train_acc, train_fscore, test_loss, test_acc, test_fscore,
                   round(time.time() - start_time, 2)))
        if (e + 1) % 10 == 0:
            print('----------best F-Score:', max(all_fscore))
            print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))
            print(confusion_matrix(best_label, best_pred, sample_weight=best_mask))
    writer.close()
