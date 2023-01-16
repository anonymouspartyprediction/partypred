import sys
import json
import pickle
import numpy as np

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

for graph_type in GRAPH_TYPES:
    data_list = []
    for split_num in range(10):
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
            GRAPH_EMBEDDING_FOLDER = GRAPH_FOLDER + f'/{split_num}' + f'/graph_embeddings_GAT'
            with open(f'{GRAPH_EMBEDDING_FOLDER}/{graph_type}_test_earlystop_acc_GAT.pkl', 'rb') as f:
                data = pickle.load(f)
        else:
            with open(f'{GRAPH_EMBEDDING_FOLDER}/{graph_type}_test_earlystop_acc.pkl', 'rb') as f:
                data = pickle.load(f)
        data_list.append(data)
    print(graph_type)
    print(round(100*np.mean(data_list),1), round(100*np.std(data_list),1))
