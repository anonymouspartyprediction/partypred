import os
import sys
import time

import json
import pandas as pd
import numpy as np
np.random.seed(0)
import operator

import torch
torch.manual_seed(0)
from torch import nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
from dgl.nn.pytorch.conv import GATConv
dgl.random.seed(42) 

import pickle

CONFIG_PATH = sys.argv[1]

with open(CONFIG_PATH, "r") as jsonfile:
    config_dict = json.load(jsonfile)

SEMI = config_dict["SEMI"]
TWO_LAYER = config_dict["TWO_LAYER"] # use two gcn layers instead of one
INCLUDE_POLITICIANS = config_dict["INCLUDE_POLITICIANS"]
if "GAT" in config_dict:
    USE_GAT = config_dict["GAT"]
else:
    print("NOT CONFIGURED TO GAT")
    exit()

GRAPH_FOLDER = config_dict["GRAPH_FOLDER"] 

GRAPH_TYPES = config_dict["GRAPH_TYPES"]

def train_gcn_pipeline(split_num):
    if SEMI:
        if TWO_LAYER:
            GRAPH_EMBEDDING_FOLDER = GRAPH_FOLDER + f'/{split_num}' + f'/graph_embeddings_semi_twolayer'
        else:
            GRAPH_EMBEDDING_FOLDER = GRAPH_FOLDER + f'/{split_num}' + f'/graph_embeddings_semi'
    else:
        if TWO_LAYER:
            GRAPH_EMBEDDING_FOLDER = GRAPH_FOLDER + f'/{split_num}' + f'/graph_embeddings_twolayer'
        else:
            GRAPH_EMBEDDING_FOLDER = GRAPH_FOLDER + f'/{split_num}' + f'/graph_embeddings'
    if USE_GAT:
        GRAPH_EMBEDDING_FOLDER = GRAPH_FOLDER + f'/{split_num}' + f'/graph_embeddings_GAT_v2'

    if not os.path.exists(GRAPH_EMBEDDING_FOLDER):
        os.makedirs(GRAPH_EMBEDDING_FOLDER)


    device = torch.device('cuda:0')



    def adjust_learning_rate(optimizer, lr, epoch):
        if epoch <= 50:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr * epoch / 50





    class Bias(nn.Module):
        def __init__(self, size):
            super().__init__()

            self.bias = nn.Parameter(torch.Tensor(size))
            self.reset_parameters()

        def reset_parameters(self):
            nn.init.zeros_(self.bias)

        def forward(self, x):
            return x + self.bias

    class GAT(nn.Module):
        def __init__(self,
                    in_feats,
                    n_classes,
                    n_hidden,
                    n_layers,
                    n_heads,
                    activation,
                    dropout=0.0,
                    attn_drop=0.0):
            super().__init__()
            self.in_feats = in_feats
            self.n_classes = n_classes
            self.n_hidden = n_hidden
            self.n_layers = n_layers
            self.num_heads = n_heads

            self.convs = nn.ModuleList()
            self.linear = nn.ModuleList()
            self.bns = nn.ModuleList()

            for i in range(n_layers):
                in_hidden = n_heads * n_hidden if i > 0 else in_feats
                out_hidden = n_hidden if i < n_layers - 1 else n_classes
                out_channels = n_heads

                self.convs.append(GATConv(in_hidden,
                                        out_hidden,
                                        num_heads=n_heads,
                                        attn_drop=attn_drop))
                self.linear.append(nn.Linear(in_hidden, out_channels * out_hidden, bias=False))
                if i < n_layers - 1:
                    self.bns.append(nn.BatchNorm1d(out_channels * out_hidden))

            self.bias_last = Bias(n_classes)

            self.dropout0 = nn.Dropout(min(0.1, dropout))
            self.dropout = nn.Dropout(dropout)
            self.activation = activation

        def forward(self, graph, feat):
            h = feat
            h = self.dropout0(h)

            for i in range(self.n_layers):
                conv = self.convs[i](graph, h)
                linear = self.linear[i](h).view(conv.shape)

                h = conv + linear

                if i < self.n_layers - 1:
                    h = h.flatten(1)
                    h = self.bns[i](h)
                    h = self.activation(h)
                    h = self.dropout(h)

            h = h.mean(1)
            h = self.bias_last(h)

            return F.log_softmax(h, dim=-1)

        def embed(self, graph, feat):
            h = feat

            for i in range(self.n_layers - 1):
                conv = self.convs[i](graph, h)
                linear = self.linear[i](h).view(conv.shape)

                h = conv + linear

                if i < self.n_layers - 2:
                    h = h.flatten(1)
                    h = self.bns[i](h)
                    h = self.activation(h)

            return h



    for graph_type in GRAPH_TYPES:

        # load graphs


        if os.path.exists(f'{GRAPH_EMBEDDING_FOLDER}/{graph_type}.pkl'): 
            continue
        try:
            if "PATH_TO_GRAPHS" in config_dict:
                with open(f'{config_dict["PATH_TO_GRAPHS"]}/{split_num}/{graph_type}.pkl', 'rb') as f:
                    save_dict = pickle.load(f)
            else:
                with open(f'{GRAPH_FOLDER}/{split_num}/{graph_type}.pkl', 'rb') as f:
                    save_dict = pickle.load(f)
            g = save_dict[f'g_{graph_type}'].to(device)
        except:
            continue

        node_to_true_label = save_dict[f'node_to_true_label_{graph_type}']
        node_to_true_label_train = save_dict[f'node_to_true_label_{graph_type}_train']
        node_to_true_label_val = save_dict[f'node_to_true_label_{graph_type}_val']
        node_to_true_label_test = save_dict[f'node_to_true_label_{graph_type}_test']
        if INCLUDE_POLITICIANS:
            node_to_true_label_politicians = save_dict[f'node_to_true_label_{graph_type}_politicians']
        node_to_weak_label = save_dict[f'node_to_weak_label_{graph_type}']
        node_to_handle = save_dict[f'node_to_id_{graph_type}']


            

        print(f'Training GCN: {graph_type}')


        node_features = torch.empty((g.number_of_nodes(),100)).to(device)
        torch.nn.init.xavier_uniform(node_features)
        party_floats_weak = []
        nid_list_weak = []
        party_floats_true = []
        nid_list_true = []
        party_floats_true_train = []
        nid_list_true_train = []
        party_floats_true_val = []
        nid_list_true_val = []
        party_floats_true_test = []
        nid_list_true_test = []
        party_floats_true_politicians = []
        nid_list_true_politicians = []
        counter_dem = 0
        counter_rep = 0


        for nid, label in node_to_true_label.items():
            if nid not in nid_list_true:
                if label == 1:
                    party_floats_true.append(0)
                    nid_list_true.append(nid)
                    counter_dem += 1
                elif label == 0:
                    party_floats_true.append(1)
                    nid_list_true.append(nid)
                    counter_rep += 1

        for nid, label in node_to_true_label_train.items():
            if nid not in nid_list_true_train:
                if label == 1:
                    party_floats_true_train.append(0)
                    nid_list_true_train.append(nid)
                    counter_dem += 1
                elif label == 0:
                    party_floats_true_train.append(1)
                    nid_list_true_train.append(nid)
                    counter_rep += 1
        for nid, label in node_to_true_label_val.items():
            if nid not in nid_list_true_val:
                if label == 1:
                    party_floats_true_val.append(0)
                    nid_list_true_val.append(nid)
                    counter_dem += 1
                elif label == 0:
                    party_floats_true_val.append(1)
                    nid_list_true_val.append(nid)
                    counter_rep += 1
        for nid, label in node_to_true_label_test.items():
            if nid not in nid_list_true_test:
                if label == 1:
                    party_floats_true_test.append(0)
                    nid_list_true_test.append(nid)
                    counter_dem += 1
                elif label == 0:
                    party_floats_true_test.append(1)
                    nid_list_true_test.append(nid)
                    counter_rep += 1

        if INCLUDE_POLITICIANS:
            for nid, label in node_to_true_label_politicians.items():
                if nid not in nid_list_true_politicians:
                    if label == 1:
                        party_floats_true_politicians.append(0)
                        nid_list_true_politicians.append(nid)
                        counter_dem += 1
                    elif label == 0:
                        party_floats_true_politicians.append(1)
                        nid_list_true_politicians.append(nid)
                        counter_rep += 1

        for nid, label in node_to_weak_label.items():
            if node_to_weak_label[nid] == 'liberal':
                party_floats_weak.append(0)
                nid_list_weak.append(nid)
                counter_dem += 1
            elif node_to_weak_label[nid] == 'conservative':
                party_floats_weak.append(1)
                nid_list_weak.append(nid)
                counter_rep += 1



        nid_list_train = nid_list_weak[1000:]
        nid_list_earlystop = nid_list_weak[:1000]
        train_nodes = torch.tensor(np.array([int(x) for x in list(nid_list_train)])).to(device) 
        earlystop_nodes = torch.tensor(np.array([int(x) for x in list(nid_list_earlystop)])).to(device) 
        train_labels = torch.nn.functional.one_hot(torch.tensor([int(x) for x in party_floats_weak[1000:]]).type(torch.int64)).float().to(device)
        earlystop_labels = torch.nn.functional.one_hot(torch.tensor([int(x) for x in party_floats_weak[:1000]]).type(torch.int64)).float().to(device)
        val_labels = torch.nn.functional.one_hot(torch.tensor([int(x) for x in party_floats_true_val]).type(torch.int64)).float().to(device)
        test_labels = torch.nn.functional.one_hot(torch.tensor([int(x) for x in party_floats_true_test]).type(torch.int64)).float().to(device)

        val_nodes = torch.tensor(np.array([int(x) for x in nid_list_true_val])).to(device)
        test_nodes = torch.tensor(np.array([int(x) for x in nid_list_true_test])).to(device)

        if INCLUDE_POLITICIANS:
            politician_nodes = torch.tensor(np.array([int(x) for x in nid_list_true_politicians])).to(device)

        print('labels size:', train_labels.size())

        print('Number of nodes:', g.number_of_nodes())
        
        model = GAT(100,2,100,3,3,activation=F.relu, dropout=0.75, attn_drop=0.05).to(device)
        gat_lr = 0.002
        opt = torch.optim.RMSprop(model.parameters(), lr=gat_lr)

        g = dgl.add_reverse_edges(g, copy_ndata=True)
        g = g.add_self_loop()

        max_earlystop_acc = 0
        
        train_labels_reduced = torch.argmax(train_labels, 1).long()
        val_labels_reduced = torch.argmax(val_labels, 1)
        test_labels_reduced = torch.argmax(test_labels, 1)
        earlystop_labels_reduced = torch.argmax(earlystop_labels, 1)

        t0 = time.time()

        for epoch in range(2000):
            adjust_learning_rate(opt, gat_lr, epoch)

            model.train()
            opt.zero_grad()

            logits = model(g, node_features)
            train_loss = F.nll_loss(logits[train_nodes],train_labels_reduced)              #   (logits[train_idx], labels.squeeze(1)[train_idx])
            train_loss.backward()

            opt.step()
            
            if epoch % 25 == 0:
                print(epoch, train_loss.item())

            model.eval()
            with torch.no_grad():
                logits = model(g, node_features)
                
                y_pred = logits.argmax(dim=-1, keepdim=False)

                val_preds = y_pred[val_nodes]
                tmp_val_acc = (val_preds == val_labels_reduced).float().sum().item() / val_labels_reduced.size(0)
                test_preds = y_pred[test_nodes]

                tmp_test_acc = (test_preds == test_labels_reduced).float().sum().item() / test_labels_reduced.size(0)


                print(f'Epoch {epoch} | Train loss: {train_loss.item():.4f} | Val acc: {tmp_val_acc:.4f} | Test acc {tmp_test_acc:.4f}')

                #if valid_acc > best_acc:
                #    best_acc = valid_acc
                #    best_model = copy.deepcopy(model)



                earlystop_preds = y_pred[earlystop_nodes]
                tmp_earlystop_acc = (earlystop_preds == earlystop_labels_reduced).float().sum().item() / earlystop_labels_reduced.size(0)
                
                if tmp_earlystop_acc > max_earlystop_acc:
                    node_embeddings = model.embed(g, node_features).detach().cpu().numpy()

                    print(f'Epoch {epoch} | Train loss: {train_loss.item():.4f} | Val acc: {tmp_val_acc:.4f} | Test acc {tmp_test_acc:.4f}')
                    test_earlystop_acc = tmp_test_acc

                    max_earlystop_acc = tmp_earlystop_acc
            with open(f'{GRAPH_EMBEDDING_FOLDER}/{graph_type}_test_earlystop_acc_GAT.pkl', 'wb') as f:
                pickle.dump(test_earlystop_acc, f)


        t1 = time.time()

        print('TIME taken:', t1 - t0)



        new_node_to_handle = {}
        embeddings_by_handle = {}
        tweet_skip_connect = {}
        handle_to_node = {}
        text_only = node_features.detach().cpu().numpy()
        for nid, name in node_to_handle.items():
                embeddings_by_handle[name] = node_embeddings[nid]
                new_node_to_handle[nid] = name
                handle_to_node[name] = nid
                tweet_skip_connect[name] = text_only[nid]

        save_dict['embeddings_by_name'] = embeddings_by_handle
        save_dict['embeddings_by_name'] = {key : value.mean(0) for key, value in save_dict['embeddings_by_name'].items()}
        save_dict['node_to_name'] = new_node_to_handle
        save_dict['node_embeddings'] = node_embeddings
        save_dict['tweet_skip_connect'] = tweet_skip_connect
        save_dict['handle_to_node'] = handle_to_node
        save_dict['train_nodes'] = train_nodes
        save_dict['val_nodes'] = val_nodes
        save_dict['test_nodes'] = test_nodes
        if INCLUDE_POLITICIANS:
            save_dict['politician_nodes'] = politician_nodes

        # save
        with open(f'{GRAPH_EMBEDDING_FOLDER}/{graph_type}.pkl', 'wb') as f:
            pickle.dump(save_dict, f)






if len(sys.argv) > 2:
    split_num = sys.argv[2]
    if len(sys.argv) > 3:
        GRAPH_TYPES = [sys.argv[3]]
    train_gcn_pipeline(split_num)
else:
    NUM_SPLITS = config_dict["NUM_SPLITS"]
    for split_num in range(NUM_SPLITS):
        train_gcn_pipeline(split_num)