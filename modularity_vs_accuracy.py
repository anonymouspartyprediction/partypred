import json
import os
import re
import glob

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from pandas._libs import missing
import numpy as np
import scipy as sp
from scipy.sparse import coo_matrix 
from scipy.sparse import csr_matrix

import networkx as nx
import dgl
import torch

import pickle
import copy

import sys

import random
random.seed(42)

CONFIG_PATH = sys.argv[1]

with open(CONFIG_PATH, "r") as jsonfile:
    config_dict = json.load(jsonfile)

with open("config/common_data_paths.json", "r") as jsonfile:
    path_dict = json.load(jsonfile)
TEMPORAL_DATA_PATH = path_dict["TEMPORAL_DATA_PATH"]
SCREEN_NAME_TO_ID_PATH = path_dict["SCREEN_NAME_TO_ID_PATH"]
TRUE_LABELS_PATH = path_dict["TRUE_LABELS_PATH"]
OLD_CLASSIFIER_LABELS_PATH = path_dict["OLD_CLASSIFIER_LABELS_PATH"] # helper for some dataset construction stuff, old labels not used

FILTER_NUM = config_dict['FILTER_NUM']
ACTIVITY_QUANTILE = config_dict['ACTIVITY_QUANTILE'] # users below this quantile of activity, without true labels, get filtered out of graph
NUM_SPLITS = config_dict['NUM_SPLITS'] # how many times to repeat the random split
INCLUDE_POLITICIANS = config_dict['INCLUDE_POLITICIANS']
DO_PROJECT = config_dict['DO_PROJECT'] # project the graph or not
WEIGHTED = config_dict['WEIGHTED']
ALL_RELATIONS_FILTER = config_dict['ALL_RELATIONS_FILTER'] # require users to have all relations (facilitating comparisons between models requiring different relations) or not


GRAPH_FOLDER = config_dict['GRAPH_FOLDER']
if not os.path.exists(GRAPH_FOLDER):
    os.makedirs(GRAPH_FOLDER)

def convert_to_64bit_indices(A):
    A.indptr = np.array(A.indptr, copy=False, dtype=np.int64)
    A.indices = np.array(A.indices, copy=False, dtype=np.int64)
    A.data = np.array(A.data, copy=False, dtype=np.int32)
    return A

def delete_from_csr(mat, row_indices=[], col_indices=[]):
    """
    Remove the rows (denoted by ``row_indices``) and columns (denoted by ``col_indices``) from the CSR sparse matrix ``mat``.
    WARNING: Indices of altered axes are reset in the returned matrix
    """
    if not isinstance(mat, csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")

    rows = []
    cols = []
    if row_indices is not []:
        rows = list(row_indices)
    if col_indices is not []:
        cols = list(col_indices)

    if len(rows) > 0 and len(cols) > 0:
        row_mask = np.ones(mat.shape[0], dtype=bool)
        row_mask[rows] = False
        col_mask = np.ones(mat.shape[1], dtype=bool)
        col_mask[cols] = False
        return mat[row_mask][:,col_mask]
    elif len(rows) > 0:
        mask = np.ones(mat.shape[0], dtype=bool)
        mask[rows] = False
        return mat[mask]
    elif len(cols) > 0:
        mask = np.ones(mat.shape[1], dtype=bool)
        mask[cols] = False
        return mat[:,mask]
    else:
        return mat



def read_relations(relation_type):
    """
    Read relations and apply any filters on fraction of relations or users in them.
    """
    df_list = []
    for month in valid_days_dict.keys():
        for day in valid_days_dict[month].split(','):
            df = pd.read_csv(f'TEMPORAL_DATA_PATH/{relation_type}_{month}-{day}.csv', dtype={'user_id':str,'tweet_id':str, 'quoted_user_id':str,'quoted_tweet_id':str,
                                                                                       'retweeted_user_id':str, 'retweeted_tweet_id':str, 'mentioned_user_id':str})
            if relation_type == 'mention':
                df = df.rename(columns={'mentioned_user_id' : 'dest', 'user_id' : 'source'})
            elif relation_type == 'retweet':
                df = df.rename(columns={'retweeted_user_id' : 'dest', 'user_id' : 'source'})
            elif relation_type == 'hashtag':
                df = df.rename(columns={'hashtag' : 'dest', 'user_id' : 'source'})
            elif relation_type == 'quote':
                df = df.rename(columns={'quoted_user_id' : 'dest', 'user_id' : 'source'})

            df_list.append(df)

    df = pd.concat(df_list)
    df = df.drop_duplicates()

    total_relations = len(df)

    df = df.sample(frac=1.0)
    df = df.sort_values('source')

    return df, total_relations

def relations_to_matrix(df, project=True):
    """
    Convert relations into projected adjacency matrix. Apply filters on required activity in terms of entries in adjacency, if any.
    """

    if project:
        df = df.loc[df.duplicated(subset='dest', keep=False), :]

        dests_raw = list(set(df.dest.tolist()))
        sources_raw = list(set(df.source.tolist()))
        dests = {dests_raw[i] : i for i in range(len(dests_raw))}
        sources = {sources_raw[i] : i for i in range(len(sources_raw))}
        df['dest_int'] = df.dest.map(dests).astype(int)
        df['source_int'] = df.source.map(sources).astype(int)

        if WEIGHTED == False:
            df['count'] = 1

        adjacency = coo_matrix((df['count'], (df['source_int'], df['dest_int'])), shape=(len(sources_raw), len(dests_raw))).tocsr()

        # note: without below, it will often give an error "RuntimeError: nnz of the result is too large"
        adjacency = convert_to_64bit_indices(adjacency)

        projected = adjacency * adjacency.T

        dense_proj = projected.todense()
        proj_nz = np.zeros((dense_proj.shape[0], dense_proj.shape[1]))
        proj_nz[dense_proj.nonzero()] = 1
        row_sum = pd.Series(list(np.array(proj_nz.sum(axis=1)).flatten()))

        # filter
        projected = projected.tocsr()
        tmp_row_indices = row_sum[row_sum > FILTER_NUM].index.astype(int)
        tmp_row_indices = tmp_row_indices.to_list()

        tmp_row_indices_negative = row_sum[row_sum <= FILTER_NUM].index.astype(int)
        tmp_row_indices_negative = tmp_row_indices_negative.to_list()

        projected = delete_from_csr(projected, row_indices=tmp_row_indices_negative, col_indices=tmp_row_indices_negative)

        return projected, sources_raw, tmp_row_indices

    else:
        print('original relations',len(df))
        df = df.loc[df.duplicated(subset=['dest'], keep=False) | df.dest.isin(df.source.tolist()), :]
        print('relations after removing singleton destinations',len(df))
        print('unique sources', len(df.source.unique()))

        indices_raw = list(set(df.dest.tolist() + df.source.tolist()))
        tmp_indices = {indices_raw[i] : i for i in range(len(indices_raw))}
        df['dest_int'] = df.dest.map(tmp_indices).astype(int)
        df['source_int'] = df.source.map(tmp_indices).astype(int)
        reverse_df = copy.copy(df)
        reverse_df['dest_int'] = df['source_int']
        reverse_df['source_int'] = df['dest_int']
        df = df.append(reverse_df).drop_duplicates()
        print('after adding reverse relations, df length:', len(df), len(df.drop_duplicates()))

        adjacency = coo_matrix((df['count'], (df['source_int'], df['dest_int'])), shape=(len(indices_raw), len(indices_raw))).tocsr()

        print('raw adjacency nnz', adjacency.nnz)

        # note: without below, it will often give an error "RuntimeError: nnz of the result is too large"
        adjacency = convert_to_64bit_indices(adjacency)

        projected = adjacency

        print(projected.shape)


        projected = projected.tocsr()

        return projected, indices_raw, range(len(indices_raw))

def filter_for_truelabels(input_df):
    input_df = collection_correction(input_df)
    output_df = input_df[input_df.source.isin(list(id_to_true.keys()))]

    return output_df



valid_days_dict = {
        'oct':'01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31',
        'nov':'01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30',  
        'dec':'01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31',
        'jan':'01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31',
        'feb':'01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28',  
    }
month_number_dict = {'oct':'10','nov':'11','dec':'12','jan':'01', 'feb':'02'}

classifier_label_df = pd.read_json(OLD_CLASSIFIER_LABELS_PATH, lines=True, orient='records', dtype={'UID':str})
screen_name_to_id = pd.read_csv(SCREEN_NAME_TO_ID_PATH,sep='\t', dtype={'user_id':str})
screen_name_to_id = pd.Series(screen_name_to_id.user_id.values,index=screen_name_to_id.screen_name).to_dict()
true_df = pd.read_json(TRUE_LABELS_PATH,orient='records',lines=True)
id_to_true = pd.Series(true_df.true_label.values,index=true_df.screen_name).to_dict()
id_to_true_tmp = {str(key) : value for key, value in id_to_true.items()}
id_to_true = {}
found_name_to_id_counter = 0
for screen_name, value in id_to_true_tmp.items():
    if screen_name in screen_name_to_id:
        id_to_true[screen_name_to_id[screen_name]] = value
        found_name_to_id_counter += 1
    else:
        continue


filter_terms = ['conservative', 'gop', 'republican', 'trump', 'liberal', 'progressive', 'democrat', 'biden', 'progressive']
filtered_df = copy.copy(classifier_label_df)
filtered_df = filtered_df[~filtered_df.text.isna()]
filtered_df = filtered_df[filtered_df.text.str.lower().str.contains('|'.join(filter_terms))]
filtered_uids = filtered_df.UID.tolist()

def collection_correction(input_df): # correct for incorrectly stemmed words in data collection
    output_df = input_df[input_df.source.isin(filtered_uids)]
    return output_df






test_id_save_dict = {}



df, _ = read_relations('retweet')

df = filter_for_truelabels(df)



projected, sources_raw, tmp_row_indices = relations_to_matrix(df, project=False)
print(f'retweet graph size', projected.shape)
print('retweet nonzeros:', projected.nnz)

g_retweet = dgl.from_scipy(projected, eweight_name='weight', idtype=torch.int32) 

filtered_sources = [sources_raw[i] for i in tmp_row_indices]
node_to_id_retweet = {i : filtered_sources[i] for i in range(len(filtered_sources))}


true_ids = [x for x in list(id_to_true.keys())]
    
node_to_true_label_retweet =  {node_id : id_to_true[uid] if (str(uid) in true_ids)  else 'NaN' for node_id, uid in node_to_id_retweet.items()}


node_lib = [x for x in node_to_true_label_retweet.keys() if node_to_true_label_retweet[x] == 1]
node_cons = [x for x in node_to_true_label_retweet.keys() if node_to_true_label_retweet[x] == 0]

print('# lib', len(node_lib), '# cons', len(node_cons))

gnx = g_retweet.to_networkx()

modularity_dict = {}

for accuracy in [1.0,0.9,0.8,0.7,0.6,0.5]:
    modularity_list = []
    for repeat in range(10):

        gnx = gnx.subgraph(node_lib+node_cons)

        print('accuracy',accuracy)

        random.shuffle(node_lib)
        random.shuffle(node_cons)
        scrambled_lib_ids = node_lib[int((1 - accuracy)*len(node_lib)):] + node_cons[:int((1 - accuracy)*len(node_cons))]
        scrambled_cons_ids = node_lib[:int((1 - accuracy)*len(node_lib))] + node_cons[int((1 - accuracy)*len(node_cons)):]

        modularity = nx.algorithms.community.quality.modularity(gnx, [scrambled_lib_ids, scrambled_cons_ids])

        modularity_list.append(modularity)

    print('average modularity',np.mean(modularity_list), '+-', np.std(modularity_list))

    modularity_dict[accuracy] = np.mean(modularity_list)

with open('results/modularity_vs_accuracy_results.pkl', 'wb') as f:
    pickle.dump(modularity_dict, f)
