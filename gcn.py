import os
import sys
import time

import json
import pandas as pd
import numpy as np
import operator

import torch
from torch import nn
import torch.nn.functional as F

import dgl
import dgl.function as fn

import random

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
    USE_GAT = False

GRAPH_FOLDER = config_dict["GRAPH_FOLDER"] 

GRAPH_TYPES = config_dict["GRAPH_TYPES"]

def train_gcn_pipeline(split_num):
    if SEMI:
        if TWO_LAYER:
            GRAPH_EMBEDDING_FOLDER = GRAPH_FOLDER + f'/{split_num}' + f'/graph_embeddings_semi_twolayer'
        else:
            GRAPH_EMBEDDING_FOLDER = GRAPH_FOLDER + f'/{split_num}' + f'/graph_embeddings_semi_timing'
    else:
        if TWO_LAYER:
            GRAPH_EMBEDDING_FOLDER = GRAPH_FOLDER + f'/{split_num}' + f'/graph_embeddings_twolayer'
        else:
            GRAPH_EMBEDDING_FOLDER = GRAPH_FOLDER + f'/{split_num}' + f'/graph_embeddings'
    if USE_GAT:
        GRAPH_EMBEDDING_FOLDER = GRAPH_FOLDER + f'/{split_num}' + f'/graph_embeddings_GAT'

    if not os.path.exists(GRAPH_EMBEDDING_FOLDER):
        os.makedirs(GRAPH_EMBEDDING_FOLDER)


    device = torch.device('cuda:0')




    # gcn adapted from https://docs.dgl.ai/guide/training-link.html
    #gcn_msg = fn.u_mul_e('h', 'weight', 'm') 
    gcn_msg = fn.copy_src(src='h', out='m') # <- uncomment for unweighted
    gcn_reduce = fn.sum(msg='m', out='h')

    class GCNLayer(nn.Module):
        def __init__(self, in_feats, out_feats):
            super(GCNLayer, self).__init__()
            self.linear = nn.Linear(in_feats, out_feats)

        def forward(self, g, feature):
            # Creating a local scope so that all the stored ndata and edata
            # (such as the `'h'` ndata below) are automatically popped out
            # when the scope exits.
            with g.local_scope():
                g.ndata['h'] = feature
                g.update_all(gcn_msg, gcn_reduce)
                h = g.ndata['h']
                return self.linear(h)

    class DotProductPredictor(nn.Module):
        def forward(self, graph, h):
            # h contains the node representations computed from the GNN defined
            # in the node classification section (Section 5.1).
            with graph.local_scope():
                graph.ndata['h'] = h
                graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
                return graph.edata['score']

    def construct_negative_graph(graph, k):
        src, dst = graph.edges()

        neg_src = src.repeat_interleave(k).type(torch.int32)
        neg_dst = torch.randint(0, graph.number_of_nodes(), (len(src) * k,)).type(torch.int32).to(device)
        neg_graph = dgl.graph((neg_src, neg_dst), num_nodes=graph.number_of_nodes())
        return neg_graph

    if SEMI:
        class Model(nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.sage = GCNLayer(in_features, out_features)
                self.pred = DotProductPredictor()
                self.node_predictor = nn.Linear(out_features, 2)
                if TWO_LAYER:
                    self.sagetwo = GCNLayer(in_features, out_features)
            def forward(self, g, neg_g, x):
                h = self.sage(g, x)
                if TWO_LAYER:
                    h = F.relu(h)
                    h = self.sagetwo(g, h)
                return self.pred(g, h), self.pred(neg_g, h), self.node_predictor(h)
    else:
        class Model(nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.sage = GCNLayer(in_features, out_features)
                self.pred = DotProductPredictor()
                if TWO_LAYER:
                    self.sagetwo = GCNLayer(in_features, out_features)
            def forward(self, g, neg_g, x):
                h = self.sage(g, x)
                if TWO_LAYER:
                    h = F.relu(h)
                    h = self.sagetwo(g, h)
                return self.pred(g, h), self.pred(neg_g, h)

    class SeededDotProductPredictor(nn.Module):
        def forward(self, h):
            # h contains the node representations computed from the GNN
            seed_vector = torch.ones_like(h)
            scores = torch.nn.functional.cosine_similarity(h, seed_vector, dim=1)
            return scores

    

    def l2_loss(scores, labels):
        return F.mse_loss(scores.flatten(), labels.flatten())


    def bce_loss(scores, labels):
        return F.binary_cross_entropy_with_logits(scores.flatten(), labels.flatten())



    if SEMI:
        def compute_loss(pos_score, neg_score, node_preds, train_nodes, train_labels):
            n_edges = pos_score.size(0) # pos_score => n_edges x 1
            neg_score = neg_score.view(n_edges, -1) # n_edges x neg_samples

            pos_labels = torch.ones_like(pos_score)
            neg_labels = torch.zeros_like(neg_score)
            combined_scores = torch.cat([pos_score, neg_score], dim=-1) # n_edges x (1+neg_samples)
            combined_labels = torch.cat([pos_labels, neg_labels], dim=-1) # ^^^

            loss = bce_loss(combined_scores, combined_labels) + bce_loss(node_preds[train_nodes], train_labels)

            return loss
    else:
        def compute_loss(pos_score, neg_score):
            n_edges = pos_score.size(0) # pos_score => n_edges x 1
            neg_score = neg_score.view(n_edges, -1) # n_edges x neg_samples

            pos_labels = torch.ones_like(pos_score)
            neg_labels = torch.zeros_like(neg_score)
            combined_scores = torch.cat([pos_score, neg_score], dim=-1) # n_edges x (1+neg_samples)
            combined_labels = torch.cat([pos_labels, neg_labels], dim=-1) # ^^^

            loss = bce_loss(combined_scores, combined_labels)

            return loss

    def sparse_eye(size):
        """
        Returns the identity matrix as a sparse matrix
        """
        indices = torch.arange(0, size).long().unsqueeze(0).expand(2, size)
        values = torch.tensor(1.0).expand(size)
        cls = getattr(torch.sparse, values.type().split(".")[-1])
        return cls(indices, values, torch.Size([size, size])) 



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


        if SEMI:
            nid_list_train = nid_list_weak[1000:]
            nid_list_earlystop = nid_list_weak[:1000]
            train_nodes = torch.tensor(np.array([int(x) for x in list(nid_list_train)])).to(device) 
            earlystop_nodes = torch.tensor(np.array([int(x) for x in list(nid_list_earlystop)])).to(device) 
            train_labels = torch.nn.functional.one_hot(torch.tensor([int(x) for x in party_floats_weak[1000:]]).type(torch.int64)).float().to(device)
            earlystop_labels = torch.nn.functional.one_hot(torch.tensor([int(x) for x in party_floats_weak[:1000]]).type(torch.int64)).float().to(device)
            val_labels = torch.nn.functional.one_hot(torch.tensor([int(x) for x in party_floats_true_val]).type(torch.int64)).float().to(device)
            test_labels = torch.nn.functional.one_hot(torch.tensor([int(x) for x in party_floats_true_test]).type(torch.int64)).float().to(device)
        else:
            train_nodes = torch.tensor(np.array([int(x) for x in list(nid_list_weak)])).to(device)
            train_labels = torch.tensor([int(x)*2 - 1 for x in party_floats_weak]).type(torch.int64).float().to(device)
            val_labels = torch.tensor([int(x)*2 - 1 for x in party_floats_true_val]).type(torch.int64).float().to(device)
            test_labels = torch.tensor([int(x)*2 - 1 for x in party_floats_true_test]).type(torch.int64).float().to(device)

        num_labels = len(party_floats_true)
        val_nodes = torch.tensor(np.array([int(x) for x in nid_list_true_val])).to(device)
        test_nodes = torch.tensor(np.array([int(x) for x in nid_list_true_test])).to(device)

        if INCLUDE_POLITICIANS:
            politician_nodes = torch.tensor(np.array([int(x) for x in nid_list_true_politicians])).to(device)

        print('labels size:', train_labels.size())
        n_features = node_features.size(1)
        print('Number of nodes:', g.number_of_nodes())
        k = 5
        model = Model(n_features, 100).to(device)
        opt = torch.optim.Adam(model.parameters())

        if SEMI:
            max_earlystop_acc = 0
            
            earlystop_labels_reduced = torch.argmax(earlystop_labels, 1)
            val_labels_reduced = torch.argmax(val_labels, 1)
            test_labels_reduced = torch.argmax(test_labels, 1)

            t0 = time.time()

            for epoch in range(1000):
                negative_graph = construct_negative_graph(g, k)
                pos_score, neg_score, node_preds = model(g, negative_graph, node_features)
                loss = compute_loss(pos_score, neg_score, node_preds, train_nodes, train_labels)
                            
                opt.zero_grad()
                loss.backward()
                opt.step()
                if epoch % 25 == 0:
                    print(epoch, loss.item())

                # Compute val acc
                with torch.no_grad():
                    all_embeddings = model.sage(g, node_features)
                    earlystop_embeddings = all_embeddings[earlystop_nodes]
                    earlystop_preds = torch.argmax(model.node_predictor(earlystop_embeddings), 1)
                    tmp_earlystop_acc = (earlystop_preds == earlystop_labels_reduced).float().sum().item() / earlystop_labels_reduced.size(0)
                    
                    if tmp_earlystop_acc > max_earlystop_acc:
                        node_embeddings = all_embeddings.detach().cpu().numpy()
                        
                        val_embeddings = all_embeddings[val_nodes]
                        val_preds = torch.argmax(model.node_predictor(val_embeddings), 1)
                        tmp_val_acc = (val_preds == val_labels_reduced).float().sum().item() / val_labels_reduced.size(0)

                        test_embeddings = all_embeddings[test_nodes]
                        test_preds = torch.argmax(model.node_predictor(test_embeddings), 1)
                        tmp_test_acc = (test_preds == test_labels_reduced).float().sum().item() / test_labels_reduced.size(0)
                        print('Epoch:',epoch,'val_acc',tmp_val_acc,'test_acc',tmp_test_acc)
                        test_earlystop_acc = tmp_test_acc
                        max_earlystop_acc = tmp_earlystop_acc

            t1 = time.time()

            print('TIME taken:', t1 - t0)

            with torch.no_grad():
                val_embeddings = model.sage(g, node_features)[val_nodes]
                val_preds = torch.argmax(model.node_predictor(val_embeddings), 1)
                val_acc = (val_preds == val_labels_reduced).float().sum().item() / val_labels_reduced.size(0)
                test_embeddings = model.sage(g, node_features)[test_nodes]
                test_preds = torch.argmax(model.node_predictor(test_embeddings), 1)
                test_acc = (test_preds == test_labels_reduced).float().sum().item() / test_labels_reduced.size(0)
                print('Final Epoch:',epoch,'val_acc',val_acc,'test_acc',test_acc)
                print(pd.Series(test_preds.cpu().numpy()).value_counts())

            with open(f'{GRAPH_EMBEDDING_FOLDER}/{graph_type}_test_earlystop_acc.pkl', 'wb') as f:
                pickle.dump(test_earlystop_acc, f)
        else:
            for epoch in range(1000):
                negative_graph = construct_negative_graph(g, k)
                pos_score, neg_score = model(g, negative_graph, node_features)
                loss = compute_loss(pos_score, neg_score)
                            
                opt.zero_grad()
                loss.backward()
                opt.step()
                if epoch % 25 == 0:
                    print(epoch, loss.item())
            node_embeddings = model.sage(g, node_features).detach().cpu().numpy()



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