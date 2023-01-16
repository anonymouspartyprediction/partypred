import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

import pandas as pd
import numpy as np
import random

import pickle
import sys
import json

from sklearn.metrics import accuracy_score, f1_score

num_layers_list = [2]
output_df = pd.DataFrame()
output_df_f1 = pd.DataFrame()
for LABELPROP_NUM_LAYERS in num_layers_list:
    LABELPROP_ALPHA = 0.5

    CONFIG_PATH = sys.argv[1]

    with open(CONFIG_PATH, "r") as jsonfile:
        config_dict = json.load(jsonfile)

    GRAPH_FOLDER = config_dict["GRAPH_FOLDER"]
    GRAPH_TYPES = config_dict["GRAPH_TYPES"]
    NUM_SPLITS = config_dict["NUM_SPLITS"]

    # IMPLEMENTATION BASED ON: https://github.com/dmlc/dgl/tree/85231dc93bc11515bea95cbc67e7347c0bfea467/examples/pytorch/label_propagation
    class LabelPropagation(nn.Module):
        r"""
        Description
        -----------
        Introduced in `Learning from Labeled and Unlabeled Data with Label Propagation <https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.14.3864&rep=rep1&type=pdf>`_
        .. math::
            \mathbf{Y}^{\prime} = \alpha \cdot \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2} \mathbf{Y} + (1 - \alpha) \mathbf{Y},
        where unlabeled data is inferred by labeled data via propagation.
        Parameters
        ----------
            num_layers: int
                The number of propagations.
            alpha: float
                The :math:`\alpha` coefficient.
        """
        def __init__(self, num_layers, alpha):
            super(LabelPropagation, self).__init__()

            self.num_layers = num_layers
            self.alpha = alpha
        
        @torch.no_grad()
        def forward(self, g, labels, mask=None, post_step=lambda y: y.clamp_(0., 1.)):
            with g.local_scope():
                if labels.dtype == torch.long:
                    labels = F.one_hot(labels.view(-1)).to(torch.float32)
                
                y = labels
                if mask is not None:
                    y = torch.zeros_like(labels)
                    y[mask] = labels[mask]
                
                last = (1 - self.alpha) * y
                degs = g.in_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5).to(labels.device).unsqueeze(1)

                for _ in range(self.num_layers):
                    # Assume the graphs to be undirected
                    g.ndata['h'] = y * norm
                    g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                    y = last + self.alpha * g.ndata.pop('h') * norm
                    y = post_step(y)
                    last = (1 - self.alpha) * y
                
                return y



    device = torch.device('cuda:0')


    output_dict = {}
    output_dict_f1 = {}

    testing_confs = [('normal','normal'), ('politician','normal'),('both','normal'),('normal','politician'),('politician','politician'),('both','politician')]
    print(testing_confs[:3])
    test_acc_dict = {key : {c_k : [] for c_k in testing_confs} for key in GRAPH_TYPES}
    test_f1_dict = {key : {c_k : [] for c_k in testing_confs} for key in GRAPH_TYPES}

    for graph_type in GRAPH_TYPES:
        print(graph_type)
        test_acc_list = []
        test_f1_list = []
        for split_num in range(NUM_SPLITS):
            with open(f'{GRAPH_FOLDER}/{split_num}/{graph_type}.pkl', 'rb') as f:
                save_dict = pickle.load(f)
            g = save_dict[f'g_{graph_type}']

            node_to_true_label = save_dict[f'node_to_true_label_{graph_type}']
            node_to_true_label_test = save_dict[f'node_to_true_label_{graph_type}_test']
            if graph_type != 'likes':
                node_to_true_label_politicians = save_dict[f'node_to_true_label_{graph_type}_politicians']
            node_to_weak_label = save_dict[f'node_to_weak_label_{graph_type}']
            node_to_handle = save_dict[f'node_to_id_{graph_type}']

            filtered_node_to_true_label = {key : value for key, value in node_to_true_label_test.items() if (value == 0) or (value == 1)}
            filtered_node_to_weak_label = {key : value for key, value in node_to_weak_label.items() if (value == 'liberal') or (value == 'conservative')}

            filtered_node_to_weak_label = {key : 0 if value == 'liberal' else 1 for key, value in filtered_node_to_weak_label.items()}

            if graph_type != 'likes':
                if config_dict["INCLUDE_POLITICIANS"]:
                    filtered_node_to_true_label_politicians = {key : value for key, value in node_to_true_label_politicians.items() if (value == 0) or (value == 1)}

                    nid_list_true_politicians = list(filtered_node_to_true_label_politicians.keys())
                    random.shuffle(nid_list_true_politicians)
                    nid_list_true_politicians_train = set(nid_list_true_politicians[:int(len(nid_list_true_politicians) * .75)])
                    nid_list_true_politicians_test = set(nid_list_true_politicians[int(len(nid_list_true_politicians) * .75):])
                    print(len(nid_list_true_politicians))



            if config_dict["INCLUDE_POLITICIANS"]:
                all_labels_dict = {testing_conf : [] for testing_conf in testing_confs}
                train_mask_dict = {testing_conf : [] for testing_conf in testing_confs}
                test_mask_dict = {testing_conf : [] for testing_conf in testing_confs}
                
                for tmp_nid in g.nodes().tolist():
                    
                    if tmp_nid in filtered_node_to_true_label:
                        for testing_conf in testing_confs[:3]:
                            all_labels_dict[testing_conf].append(1 - node_to_true_label[tmp_nid])
                            train_mask_dict[testing_conf].append(0)
                            test_mask_dict[testing_conf].append(1)
                        for testing_conf in testing_confs[3:]:
                            all_labels_dict[testing_conf].append(0)
                            train_mask_dict[testing_conf].append(0)
                            test_mask_dict[testing_conf].append(0)
                    
                    elif tmp_nid in filtered_node_to_weak_label:
                        for testing_conf in [testing_confs[0], testing_confs[2], testing_confs[3], testing_confs[5]]:
                            all_labels_dict[testing_conf].append(filtered_node_to_weak_label[tmp_nid])
                            train_mask_dict[testing_conf].append(1)
                            test_mask_dict[testing_conf].append(0)
                        for testing_conf in [testing_confs[1], testing_confs[4]]:
                            all_labels_dict[testing_conf].append(0)
                            train_mask_dict[testing_conf].append(0)
                            test_mask_dict[testing_conf].append(0)

                    elif tmp_nid in nid_list_true_politicians_test:
                        for testing_conf in testing_confs[3:]:
                            all_labels_dict[testing_conf].append(1 - node_to_true_label_politicians[tmp_nid])
                            train_mask_dict[testing_conf].append(0)
                            test_mask_dict[testing_conf].append(1)
                        for testing_conf in testing_confs[1:3]:
                            all_labels_dict[testing_conf].append(1 - node_to_true_label_politicians[tmp_nid])
                            train_mask_dict[testing_conf].append(1)
                            test_mask_dict[testing_conf].append(0)
                        for testing_conf in [testing_confs[0]]:
                            all_labels_dict[testing_conf].append(0)
                            train_mask_dict[testing_conf].append(0)
                            test_mask_dict[testing_conf].append(0)

                    elif tmp_nid in nid_list_true_politicians_train:
                        for testing_conf in testing_confs[4:]:
                            all_labels_dict[testing_conf].append(1 - node_to_true_label_politicians[tmp_nid])
                            train_mask_dict[testing_conf].append(1)
                            test_mask_dict[testing_conf].append(0)
                        for testing_conf in testing_confs[1:3]:
                            all_labels_dict[testing_conf].append(1 - node_to_true_label_politicians[tmp_nid])
                            train_mask_dict[testing_conf].append(1)
                            test_mask_dict[testing_conf].append(0)
                        for testing_conf in [testing_confs[0], testing_confs[3]]:
                            all_labels_dict[testing_conf].append(0)
                            train_mask_dict[testing_conf].append(0)
                            test_mask_dict[testing_conf].append(0)

                    else:
                        for testing_conf in testing_confs:
                            all_labels_dict[testing_conf].append(0)
                            train_mask_dict[testing_conf].append(0)
                            test_mask_dict[testing_conf].append(0)
                            


                for testing_conf in testing_confs:
                    all_labels_list = all_labels_dict[testing_conf]
                    train_mask_list = train_mask_dict[testing_conf]
                    test_mask_list = test_mask_dict[testing_conf]
                    all_labels_list = [1 - x for x in all_labels_list]
                    labels = torch.tensor(all_labels_list).to(device).long()
                    train_mask = torch.tensor(train_mask_list).to(device).bool()
                    test_mask = torch.tensor(test_mask_list).to(device).bool()

                    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze().to(device)
                    test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze().to(device)

                    g = g.to(device)

                    # label propagation
                    lp = LabelPropagation(LABELPROP_NUM_LAYERS, LABELPROP_ALPHA)
                    logits = lp(g, labels, mask=train_idx)

                    preds = logits[test_idx].argmax(dim=1).cpu().numpy()
                    true = labels[test_idx].cpu().numpy()

                    test_acc = accuracy_score(true,preds)
                    test_f1 = f1_score(true,preds, average='macro')
                    print(testing_conf, "Test Acc {:.4f}".format(test_acc))
                    test_acc_dict[graph_type][testing_conf].append(100*test_acc)
                    test_f1_dict[graph_type][testing_conf].append(100*test_f1)

            else:
                
                all_labels_list = []
                train_mask_list = []
                test_mask_list = []
                for tmp_nid in g.nodes().tolist():
                    if tmp_nid in nid_list_true:
                        all_labels_list.append(1 - node_to_true_label[tmp_nid])

                        if tmp_nid in nid_list_true_test:
                            train_mask_list.append(0)
                            test_mask_list.append(1)
                        else:
                            train_mask_list.append(0)
                            test_mask_list.append(0)

                    elif tmp_nid in nid_list_weak:
                        if node_to_weak_label[tmp_nid] == 'liberal':
                            all_labels_list.append(0)
                        elif node_to_weak_label[tmp_nid] == 'conservative':
                            all_labels_list.append(1)
                        
                        
                        train_mask_list.append(1)
                        test_mask_list.append(0)

                    else:
                        all_labels_list.append(0)
                        train_mask_list.append(0)
                        test_mask_list.append(0)

                all_labels_list = [1 - x for x in all_labels_list]
                labels = torch.tensor(all_labels_list).to(device).long()
                train_mask = torch.tensor(train_mask_list).to(device).bool()
                test_mask = torch.tensor(test_mask_list).to(device).bool()

                train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze().to(device)
                test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze().to(device)

                g = g.to(device)

                # label propagation
                lp = LabelPropagation(LABELPROP_NUM_LAYERS, LABELPROP_ALPHA)
                logits = lp(g, labels, mask=train_idx)

                preds = logits[test_idx].argmax(dim=1).cpu().numpy()
                true = labels[test_idx].cpu().numpy()

                test_acc = accuracy_score(true,preds)
                test_f1 = f1_score(true,preds, average='macro')
                print("Test Acc {:.4f}".format(test_acc))
                test_acc_list.append(test_acc)
                test_f1_list.append(test_f1)

        test_acc_list = [x*100 for x in test_acc_list]
        test_f1_list = [x*100 for x in test_f1_list]
        output_dict[graph_type] = str(round(np.mean(test_acc_list),1)) + '+-' + str(round(np.std(test_acc_list),1))
        output_dict_f1[graph_type] = str(round(np.mean(test_f1_list),1)) + '+-' + str(round(np.std(test_f1_list),1))
        print(output_dict[graph_type])
        print(output_dict_f1[graph_type])

    if config_dict["INCLUDE_POLITICIANS"]:
        for graph_type in GRAPH_TYPES:
            for testing_conf in testing_confs:
                tmp_df = pd.DataFrame([[str(round(np.mean(test_acc_dict[graph_type][testing_conf]),1)) + '+-' + str(round(np.std(test_acc_dict[graph_type][testing_conf]),1))]], index=[graph_type + ' ' + testing_conf[0] + ' ' + testing_conf[1]])
                output_df = pd.concat([output_df, tmp_df])
                tmp_df_f1 = pd.DataFrame([[str(np.mean(test_f1_dict[graph_type][testing_conf])) + '+-' + str(np.std(test_f1_dict[graph_type][testing_conf]))]], index=[graph_type + ' ' + testing_conf[0] + ' ' + testing_conf[1]])
                output_df_f1 = pd.concat([output_df_f1, tmp_df_f1])

        partial_path_to_save = CONFIG_PATH.split('/')[-1].split('.json')[0]
        output_df.to_csv(f'results/{partial_path_to_save}_labelprop.csv', index=True)
        output_df_f1.to_csv(f'results/{partial_path_to_save}_labelprop_f1.csv', index=True)
    else:
        tmp_df = pd.DataFrame(output_dict, index=[LABELPROP_NUM_LAYERS])
        output_df = pd.concat([output_df, tmp_df])
        tmp_df_f1 = pd.DataFrame(output_dict_f1, index=[LABELPROP_NUM_LAYERS])
        output_df_f1 = pd.concat([output_df_f1, tmp_df_f1])

        partial_path_to_save = CONFIG_PATH.split('/')[-1].split('.json')[0]
        output_df.to_csv(f'results/{partial_path_to_save}_labelprop.csv', index=False)
        output_df_f1.to_csv(f'results/{partial_path_to_save}_labelprop_f1.csv', index=False)
