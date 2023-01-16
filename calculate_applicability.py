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

import dgl
import torch

import pickle
import copy

import sys

import random
random.seed(42)

CONFIG_PATH = 'config/no_allrelations_filter.json'

with open(CONFIG_PATH, "r") as jsonfile:
    config_dict = json.load(jsonfile)

FILTER_NUM = config_dict['FILTER_NUM']
ACTIVITY_QUANTILE = config_dict['ACTIVITY_QUANTILE'] # users below this quantile of activity, without true labels, get filtered out of graph
NUM_SPLITS = config_dict['NUM_SPLITS'] # how many times to repeat the random split
INCLUDE_POLITICIANS = config_dict['INCLUDE_POLITICIANS']
DO_PROJECT = config_dict['DO_PROJECT'] # project the graph or not
WEIGHTED = config_dict['WEIGHTED']
ALL_RELATIONS_FILTER = config_dict['ALL_RELATIONS_FILTER'] # require users to have all relations (facilitating comparisons between models requiring different relations) or not


with open("config/common_data_paths.json", "r") as jsonfile:
    path_dict = json.load(jsonfile)

POLITICIANS_FOLDER = path_dict['POLITICIANS_FOLDER'] # path to politicians data folder
FRIENDS_FOLDER = path_dict['FRIENDS_FOLDER'] # path to friends data folder
FOLLOWERS_FOLDER = path_dict['FOLLOWERS_FOLDER'] # path to followers data folder
POLITICIAN_FRIENDS_FOLDER = path_dict['POLITICIAN_FRIENDS_FOLDER']
POLITICIAN_FOLLOWERS_FOLDER = path_dict['POLITICIAN_FOLLOWERS_FOLDER']
LIKES_PATH = path_dict['LIKES_PATH'] # path to likes data file

TEMPORAL_DATA_PATH = path_dict["TEMPORAL_DATA_PATH"]
OLD_CLASSIFIER_LABELS_PATH = path_dict["OLD_CLASSIFIER_LABELS_PATH"] # helper for some dataset construction stuff, old labels not used
PROFILE_CLASSIFIER_USERS_PATH = path_dict["PROFILE_CLASSIFIER_USERS_PATH"]
SCREEN_NAME_TO_ID_PATH = path_dict["SCREEN_NAME_TO_ID_PATH"]
TRUE_LABELS_PATH = path_dict["TRUE_LABELS_PATH"]
POLITICIANS_FOLDER = path_dict['POLITICIANS_FOLDER'] # path to politicians data folder

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

def read_likes():
    df = pd.read_json(LIKES_PATH,lines=True,orient='records', dtype={'dest':str,'source':str,'dest_author_id':str})

    df = df.groupby(['source','dest']).size().reset_index(name='count')

    total_relations = len(df)

    df = df.sample(frac=1.0)
    df = df.sort_values('source')

    print(len(df), len(list(df.source.unique())), df.groupby(['source']).sum()["count"].max())

    return df, total_relations

def read_friends(include_politicians=INCLUDE_POLITICIANS):

    file_list = glob.glob(FRIENDS_FOLDER)

    df_list = []
    len_list = []

    for fname in file_list:
        if 'error_users' in fname:
            continue
        if 'capped_users' in fname:
            continue
        with open(fname,'r') as f:
            file_string_data = f.read()
            if len(file_string_data) > 0:
                try:
                    tmp_follower = json.loads(file_string_data, parse_int=(lambda x: str(x)))
                except:
                    print(fname)
                    print(file_string_data)
        tmp_source = fname.split('/')[-1].split('.')[0]
        tmp_df = pd.DataFrame()
        tmp_df['dest'] = tmp_follower
        tmp_df['source'] = tmp_source
        tmp_df['count'] = 1
        df_list.append(tmp_df)
        len_list.append(len(tmp_df))

    full_df = pd.concat(df_list)

    print('average friends per user:', np.mean(len_list))
    print('total friends for non-politicians', np.sum(len_list))
    print('max friends for non-politicians', np.max(len_list))
    print('unique friends', len(full_df.dest.unique()))

    if include_politicians:

        file_list = glob.glob(POLITICIAN_FRIENDS_FOLDER)

        df_list = []
        len_list = []

        unmatched_politicians = 0
        matched_politicians = 0

        lower_screen_name_to_id_congress_users = {key.lower():value for key, value in screen_name_to_id_congress_users.items()}
        for fname in file_list:
            if 'error_users' in fname:
                continue
            if 'capped_users' in fname:
                continue
            with open(fname,'r') as f:
                file_string_data = f.read()
                if len(file_string_data) > 0:
                    try:
                        tmp_follower = json.loads(file_string_data, parse_int=(lambda x: str(x)))
                    except:
                        print(fname)
                        print(file_string_data)
            tmp_source = fname.split('/')[-1].split('.')[0]
            if tmp_source in screen_name_to_id_congress_users:
                tmp_source = screen_name_to_id_congress_users[tmp_source]
            elif tmp_source.lower() in lower_screen_name_to_id_congress_users:
                tmp_source = lower_screen_name_to_id_congress_users[tmp_source.lower()]
            else:
                unmatched_politicians += 1
                continue
            matched_politicians += 1
            tmp_df = pd.DataFrame()
            tmp_df['dest'] = tmp_follower
            tmp_df['source'] = tmp_source
            tmp_df['count'] = 1
            df_list.append(tmp_df)
            len_list.append(len(tmp_df))

        print('UNMATCHED POLITICIANS TOTAL:', unmatched_politicians, 'MATCHED TOTAL:', matched_politicians)
        tmp_politicians_df = pd.concat(df_list)
        print('average friends per politician:', np.mean(len_list), '+-', np.std(len_list), 'total', np.sum(len_list), 'max', np.max(len_list), '# with at least one:', len([x for x in len_list if x > 0]), 'out of total', len(len_list))
        print('unique politician friends', len(tmp_politicians_df.dest.unique()))
        full_df = full_df.append(tmp_politicians_df)

    total_relations = len(full_df)

    return full_df, total_relations

def read_followers(include_politicians=INCLUDE_POLITICIANS):

    
    file_list = glob.glob(FOLLOWERS_FOLDER)

    df_list = []
    len_list = []

    for fname in file_list:
        if 'error_users' in fname:
            continue
        if 'capped_users' in fname:
            continue
        with open(fname,'r') as f:
            file_string_data = f.read()
            if len(file_string_data) > 0:
                try:
                    tmp_follower = json.loads(file_string_data, parse_int=(lambda x: str(x)))
                except:
                    print(fname)
                    print(file_string_data)
        tmp_source = fname.split('/')[-1].split('.')[0]
        tmp_df = pd.DataFrame()
        tmp_df['dest'] = tmp_follower
        tmp_df['source'] = tmp_source
        tmp_df['count'] = 1
        df_list.append(tmp_df)
        len_list.append(len(tmp_df))

    full_df = pd.concat(df_list)

    print('average followers per user:', np.mean(len_list))
    print('total followers for non-politicians', np.sum(len_list))
    print('max followers for non-politicians', np.max(len_list))
    print('unique followers', len(full_df.dest.unique()))


    if include_politicians:

        file_list = glob.glob(POLITICIAN_FOLLOWERS_FOLDER)

        df_list = []
        len_list = []

        unmatched_politicians = 0
        matched_politicians = 0

        lower_screen_name_to_id_congress_users = {key.lower():value for key, value in screen_name_to_id_congress_users.items()}
        for fname in file_list:
            if 'error_users' in fname:
                continue
            if 'capped_users' in fname:
                continue
            with open(fname,'r') as f:
                file_string_data = f.read()
                if len(file_string_data) > 0:
                    try:
                        tmp_follower = json.loads(file_string_data, parse_int=(lambda x: str(x)))
                    except:
                        print(fname)
                        print(file_string_data)
            tmp_source = fname.split('/')[-1].split('.')[0]
            if tmp_source in screen_name_to_id_congress_users:
                tmp_source = screen_name_to_id_congress_users[tmp_source]
            elif tmp_source.lower() in lower_screen_name_to_id_congress_users:
                tmp_source = lower_screen_name_to_id_congress_users[tmp_source.lower()]
            else:
                unmatched_politicians += 1
                continue
            matched_politicians += 1
            tmp_df = pd.DataFrame()
            tmp_df['dest'] = tmp_follower
            tmp_df['source'] = tmp_source
            tmp_df['count'] = 1
            df_list.append(tmp_df)
            len_list.append(len(tmp_df))

        print('UNMATCHED POLITICIANS TOTAL:', unmatched_politicians, 'MATCHED TOTAL:', matched_politicians)
        tmp_politicians_df = pd.concat(df_list)
        print('average followers per politician:', np.mean(len_list), '+-', np.std(len_list), 'total', np.sum(len_list), 'max', np.max(len_list), '# with at least one:', len([x for x in len_list if x > 0]), 'out of total', len(len_list))
        print('unique politician followers', len(tmp_politicians_df.dest.unique()))
        full_df = full_df.append(tmp_politicians_df)

    total_relations = len(full_df)

    return full_df, total_relations

def read_relations(relation_type):
    """
    Read relations and apply any filters on fraction of relations or users in them.
    """
    df_list = []
    for month in valid_days_dict.keys():
        for day in valid_days_dict[month].split(','):
            df = pd.read_csv(f'{TEMPORAL_DATA_PATH}/{relation_type}_{month}-{day}.csv', dtype={'user_id':str,'tweet_id':str, 'quoted_user_id':str,'quoted_tweet_id':str,
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

    total_relations = df['count'].sum()

    #df = df.drop_duplicates()
    #total_relations = len(df)

    df = df.sample(frac=1.0)
    df = df.sort_values('source')

    return df, total_relations

def relations_to_matrix(df, project=True):
    """
    Convert relations into projected adjacency matrix. Apply filters on required activity in terms of entries in adjacency, if any.
    """
    
    df = df.loc[df.duplicated(subset='dest', keep=False), :]

    if project:
        dests_raw = list(set(df.dest.tolist()))
        sources_raw = list(set(df.source.tolist()))
        dests = {dests_raw[i] : i for i in range(len(dests_raw))}
        sources = {sources_raw[i] : i for i in range(len(sources_raw))}
        df['dest_int'] = df.dest.map(dests).astype(int)
        df['source_int'] = df.source.map(sources).astype(int)

        if WEIGHTED == False:
            df['count'] = 1

        adjacency = coo_matrix((df['count'], (df['source_int'], df['dest_int'])), shape=(len(sources_raw), len(dests_raw))).tocsr()

        # note: without below, it can give an error "RuntimeError: nnz of the result is too large"
        adjacency = convert_to_64bit_indices(adjacency)

        projected = adjacency * adjacency.T

        dense_proj = projected.todense()
        proj_nz = np.zeros((dense_proj.shape[0], dense_proj.shape[1]))
        proj_nz[dense_proj.nonzero()] = 1
        row_sum = pd.Series(list(np.array(proj_nz.sum(axis=1)).flatten()))

        # filter
        projected = projected.tocsr()
        if WEIGHTED:
            projected.data = np.log(projected.data + 1) 
        tmp_row_indices = row_sum[row_sum > FILTER_NUM].index.astype(int)
        tmp_row_indices = tmp_row_indices.to_list()

        tmp_row_indices_negative = row_sum[row_sum <= FILTER_NUM].index.astype(int)
        tmp_row_indices_negative = tmp_row_indices_negative.to_list()

        projected = delete_from_csr(projected, row_indices=tmp_row_indices_negative, col_indices=tmp_row_indices_negative)

        return projected, sources_raw, tmp_row_indices

    else:
        indices_raw = list(set(df.dest.tolist() + df.source.tolist()))
        tmp_indices = {indices_raw[i] : i for i in range(len(indices_raw))}
        df['dest_int'] = df.dest.map(tmp_indices).astype(int)
        df['source_int'] = df.source.map(tmp_indices).astype(int)
        reverse_df = copy.copy(df)
        reverse_df['dest_int'] = df['source_int']
        reverse_df['source_int'] = df['dest_int']
        df = df.append(reverse_df).drop_duplicates()
        print('after adding reverse relations, df length:', len(df), len(df.drop_duplicates()))
        
        if WEIGHTED == False:
            df['count'] = 1

        adjacency = coo_matrix((df['count'], (df['source_int'], df['dest_int'])), shape=(len(indices_raw), len(indices_raw))).tocsr()

        print('raw adjacency nnz', adjacency.nnz)

        # note: without below, it can give an error "RuntimeError: nnz of the result is too large"
        adjacency = convert_to_64bit_indices(adjacency)

        projected = adjacency

        print(projected.shape)

        dense_proj = projected.todense()
        proj_nz = np.zeros((dense_proj.shape[0], dense_proj.shape[1]))
        proj_nz[dense_proj.nonzero()] = 1
        row_sum = pd.Series(list(np.array(proj_nz.sum(axis=1)).flatten()))

        # filter
        projected = projected.tocsr()
        if WEIGHTED:
            projected.data = np.log(projected.data + 1) 

        return projected, indices_raw, range(len(indices_raw))

def activity_filter(input_df):
    input_df = collection_correction(input_df)
    print('total relations after correction:', input_df["count"].sum())
    print('total unique relations after correction:', len(input_df["dest"].unique()))
    df_user_counts = input_df[~input_df.source.isin(list(id_to_true.keys()))]
    df_user_counts = df_user_counts.groupby('source').sum().reset_index()
    df_users_lowactivity = df_user_counts[df_user_counts['count'] < df_user_counts['count'].quantile(ACTIVITY_QUANTILE)]
    df_users_lowactivity_list = df_users_lowactivity.source.tolist()
    output_df = input_df[~input_df.source.isin(df_users_lowactivity_list)]

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
id_to_weak = pd.Series(classifier_label_df.classifier_label.values,index=classifier_label_df.screen_name).to_dict()
id_to_weak_tmp = {str(key) : value for key, value in id_to_weak.items()}
id_to_weak = {}
found_name_to_id_counter = 0
for screen_name, value in id_to_weak_tmp.items():
    if screen_name in screen_name_to_id:
        id_to_weak[screen_name_to_id[screen_name]] = value
        found_name_to_id_counter += 1
    else:
        continue

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

filtered_uids = filtered_uids + list(id_to_true.keys())

print('total public users with correct profile', len(filtered_uids))


def collection_correction(input_df): # correct for incorrectly stemmed words in data collection
    if INCLUDE_POLITICIANS:
        output_df = input_df[(input_df.source.isin(filtered_uids)) | (input_df.source.isin(congress_users_ids))]
    else:
        output_df = input_df[input_df.source.isin(filtered_uids)]
    return output_df





congress_users_df = pd.read_json(f'{POLITICIANS_FOLDER}/congress_dataset_V2/congress_users.json', lines=True, orient='records')
with open(f'{POLITICIANS_FOLDER}/congress_dataset_V2/congress_users.json', 'r') as f:
    lines = f.readlines()
print('len congress users',len(congress_users_df))

screen_name_to_id_congress_users = {}
for line in lines:
    tmp_loaded = json.loads(line)
    screen_name_to_id_congress_users[tmp_loaded['json']['screen_name']] = tmp_loaded['json']['id_str']

classifier_label_df_congress_users = pd.read_json('data/politicians_users_with_classifierlabel.jsonl', lines=True, orient='records')
congress_users_screenname_to_weak = pd.Series(classifier_label_df_congress_users.classifier_label.values,index=classifier_label_df_congress_users.screen_name).to_dict()
congress_users_screenname_to_weak = {str(key) : value for key, value in congress_users_screenname_to_weak.items()}
congress_users_id_to_weak = {}
for key, value in congress_users_screenname_to_weak.items():
    congress_users_id_to_weak[screen_name_to_id_congress_users[key]] = value
id_to_weak.update(congress_users_id_to_weak)

classifier_label_df_congress_users.true_label = classifier_label_df_congress_users.true_label.map({'Democratic':1,'Republican':0})
true_label_df_congress_users = classifier_label_df_congress_users[~classifier_label_df_congress_users.true_label.isna()]
screenname_to_true_congress_users = pd.Series(true_label_df_congress_users.true_label.values,index=true_label_df_congress_users.screen_name).to_dict()
screenname_to_true_congress_users = {str(key) : value for key, value in screenname_to_true_congress_users.items()}
congress_users_id_to_true = {}
for key, value in screenname_to_true_congress_users.items():
    congress_users_id_to_true[screen_name_to_id_congress_users[key]] = value

def apply_screenname_to_id_congress_users(screenname):
    if screenname not in screen_name_to_id_congress_users:
        print('USER NOT FOUND:', screenname)
        return 'NOT_FOUND'
    else:
        return screen_name_to_id_congress_users[screenname]







print('Loading data')

congress_users_id_true_labeled = list(congress_users_id_to_true.keys())
print('total congress users with true label', len(congress_users_id_true_labeled))

source_list = []
dest_list = []
with open(f'{POLITICIANS_FOLDER}/congress_dataset_V2/congress_mention_relations.json','r') as f:
    lines = f.readlines()
for line in lines:
    tmp_loaded = json.loads(line)
    source_list.append(tmp_loaded['user_id'])
    dest_list.append(tmp_loaded['mentioned_user_id'])
mention_df_congress_users = pd.DataFrame({'source':source_list, 'dest':dest_list})
mentions_filtered_congress_users = mention_df_congress_users.groupby(['source','dest']).size().reset_index().rename(columns={0:'count'})
print('unique source in politician mentions data:', len(mentions_filtered_congress_users.source.unique()))
grouped_df = mentions_filtered_congress_users.groupby('source')['count'].sum()
print(grouped_df.mean())
mentions_filtered_congress_users = mentions_filtered_congress_users[mentions_filtered_congress_users.source.isin(congress_users_id_true_labeled)]
print('unique source in politician mentions data, filtered for true label:', len(mentions_filtered_congress_users.source.unique()))
grouped_df = mentions_filtered_congress_users.groupby('source')['count'].sum()
print(grouped_df.mean())

source_list = []
dest_list = []
with open(f'{POLITICIANS_FOLDER}/congress_dataset_V2/congress_retweet_relations.json','r') as f:
    lines = f.readlines()
for line in lines:
    tmp_loaded = json.loads(line)
    source_list.append(tmp_loaded['user_id'])
    dest_list.append(tmp_loaded['retweeted_user_id'])
retweet_df_congress_users = pd.DataFrame({'source':source_list, 'dest':dest_list})
retweet_filtered_congress_users = retweet_df_congress_users.groupby(['source','dest']).size().reset_index().rename(columns={0:'count'})
print('unique source in politician retweet data:', len(retweet_filtered_congress_users.source.unique()))
grouped_df = retweet_filtered_congress_users.groupby('source')['count'].sum()
print(grouped_df.mean())
retweet_filtered_congress_users = retweet_filtered_congress_users[retweet_filtered_congress_users.source.isin(congress_users_id_true_labeled)]
print('unique source in politician retweet data, filtered for true label:', len(retweet_filtered_congress_users.source.unique()))
grouped_df = retweet_filtered_congress_users.groupby('source')['count'].sum()
print(grouped_df.mean())

source_list = []
dest_list = []
with open(f'{POLITICIANS_FOLDER}/congress_dataset_V2/congress_hashtag_relations.json','r') as f:
    lines = f.readlines()
for line in lines:
    tmp_loaded = json.loads(line)
    source_list.append(tmp_loaded['user_id'])
    dest_list.append(tmp_loaded['hashtag'])
hashtag_df_congress_users = pd.DataFrame({'source':source_list, 'dest':dest_list})
hashtag_filtered_congress_users = hashtag_df_congress_users.groupby(['source','dest']).size().reset_index().rename(columns={0:'count'})
print('unique source in politician hashtag data:', len(hashtag_filtered_congress_users.source.unique()))
grouped_df = hashtag_filtered_congress_users.groupby('source')['count'].sum()
print(grouped_df.mean())
hashtag_filtered_congress_users = hashtag_filtered_congress_users[hashtag_filtered_congress_users.source.isin(congress_users_id_true_labeled)]
print('unique source in politician hashtag data, filtered for true label:', len(hashtag_filtered_congress_users.source.unique()))
grouped_df = hashtag_filtered_congress_users.groupby('source')['count'].sum()
print(grouped_df.mean())

source_list = []
dest_list = []
with open(f'{POLITICIANS_FOLDER}/congress_dataset_V2/congress_quote_relations.json','r') as f:
    lines = f.readlines()
for line in lines:
    tmp_loaded = json.loads(line)
    source_list.append(tmp_loaded['user_id'])
    dest_list.append(tmp_loaded['quoted_user_id'])
quote_df_congress_users = pd.DataFrame({'source':source_list, 'dest':dest_list})
quote_filtered_congress_users = quote_df_congress_users.groupby(['source','dest']).size().reset_index().rename(columns={0:'count'})
print('unique source in politician quote data:', len(quote_filtered_congress_users.source.unique()))
grouped_df = quote_filtered_congress_users.groupby('source')['count'].sum()
print(grouped_df.mean())
quote_filtered_congress_users = quote_filtered_congress_users[quote_filtered_congress_users.source.isin(congress_users_id_true_labeled)]
print('unique source in politician hashtag data, filtered for true label:', len(quote_filtered_congress_users.source.unique()))
grouped_df = quote_filtered_congress_users.groupby('source')['count'].sum()
print(grouped_df.mean())


test_id_save_dict = {}

for split_num in range(NUM_SPLITS):
    print(split_num)

    
    df, _ = read_likes()

    df = activity_filter(df)

    projected, sources_raw, tmp_row_indices = relations_to_matrix(df, project=DO_PROJECT)
    print(f'likes graph size', projected.shape)
    print('likes nonzeros:', projected.nnz)

    g_likes = dgl.from_scipy(projected, eweight_name='weight', idtype=torch.int32)

    filtered_sources = [sources_raw[i] for i in tmp_row_indices]
    node_to_id_likes = {i : filtered_sources[i] for i in range(len(filtered_sources))}


    df, _ = read_friends(include_politicians=INCLUDE_POLITICIANS)

    df = activity_filter(df)

    projected, sources_raw, tmp_row_indices = relations_to_matrix(df, project=DO_PROJECT)
    print(f'friends graph size', projected.shape)
    print('friends nonzeros:', projected.nnz)

    g_friend = dgl.from_scipy(projected, eweight_name='weight', idtype=torch.int32)

    filtered_sources = [sources_raw[i] for i in tmp_row_indices]
    node_to_id_friend = {i : filtered_sources[i] for i in range(len(filtered_sources))}




    df, _ = read_followers(include_politicians=INCLUDE_POLITICIANS)

    df = activity_filter(df)

    projected, sources_raw, tmp_row_indices = relations_to_matrix(df, project=DO_PROJECT)
    print(f'followers graph size', projected.shape)
    print('followers nonzeros:', projected.nnz)

    g_follow = dgl.from_scipy(projected, eweight_name='weight', idtype=torch.int32)

    filtered_sources = [sources_raw[i] for i in tmp_row_indices]
    node_to_id_follow = {i : filtered_sources[i] for i in range(len(filtered_sources))}
    




    df, total_relations = read_relations('mention')

    print('TOTAL',total_relations)

    df = activity_filter(df)

    if INCLUDE_POLITICIANS:
        df = df.append(mentions_filtered_congress_users)



    projected, sources_raw, tmp_row_indices = relations_to_matrix(df, project=DO_PROJECT)
    print(f'mention graph size', projected.shape)
    print('mention nonzeros:', projected.nnz)

    g_mention = dgl.from_scipy(projected, eweight_name='weight', idtype=torch.int32)

    filtered_sources = [sources_raw[i] for i in tmp_row_indices]
    node_to_id_mention = {i : filtered_sources[i] for i in range(len(filtered_sources))}



    df, _ = read_relations('retweet')

    df = activity_filter(df)

    if INCLUDE_POLITICIANS:
        df = df.append(retweet_filtered_congress_users)


    projected, sources_raw, tmp_row_indices = relations_to_matrix(df, project=DO_PROJECT)
    print(f'retweet graph size', projected.shape)
    print('retweet nonzeros:', projected.nnz)

    g_retweet = dgl.from_scipy(projected, eweight_name='weight', idtype=torch.int32) 

    filtered_sources = [sources_raw[i] for i in tmp_row_indices]
    node_to_id_retweet = {i : filtered_sources[i] for i in range(len(filtered_sources))}



    df, _ = read_relations('hashtag')

    df = activity_filter(df)

    if INCLUDE_POLITICIANS:
        df = df.append(hashtag_filtered_congress_users)


    projected, sources_raw, tmp_row_indices = relations_to_matrix(df, project=DO_PROJECT)
    print(f'hashtag graph size', projected.shape)
    print('hashtag nonzeros:', projected.nnz)

    g_hashtag = dgl.from_scipy(projected, eweight_name='weight', idtype=torch.int32) 

    filtered_sources = [sources_raw[i] for i in tmp_row_indices]
    node_to_id_hashtag = {i : filtered_sources[i] for i in range(len(filtered_sources))}



    df, _ = read_relations('quote')

    df = activity_filter(df)

    if INCLUDE_POLITICIANS:
        df = df.append(quote_filtered_congress_users)

    projected, sources_raw, tmp_row_indices = relations_to_matrix(df, project=DO_PROJECT)
    print(f'quote graph size', projected.shape)
    print('quote nonzeros:', projected.nnz)

    g_quote = dgl.from_scipy(projected, eweight_name='weight', idtype=torch.int32)

    filtered_sources = [sources_raw[i] for i in tmp_row_indices]
    node_to_id_quote = {i : filtered_sources[i] for i in range(len(filtered_sources))}






    if ALL_RELATIONS_FILTER:
        print('NOT CONFIGURED')

    else: # No all relations filter
        union_ids = set(node_to_id_mention.values()) | set(node_to_id_retweet.values()) | set(node_to_id_hashtag.values()) | set(node_to_id_quote.values()) | set(node_to_id_follow.values()) | set(node_to_id_friend.values()) | set(node_to_id_likes.values())
        union_ids_true = union_ids.intersection(set(id_to_true.keys())) - set(congress_users_id_to_true.keys())
        total = len(union_ids_true)
        print('total union ids:', total)

        for graphs_list in config_dict["GRAPHS_TO_USE_LIST"]:
            tmp_union = set()
            if 'mention' in graphs_list:
                tmp_union = tmp_union | set(node_to_id_mention.values())
            if 'retweet' in graphs_list:
                tmp_union = tmp_union | set(node_to_id_retweet.values())
            if 'hashtag' in graphs_list:
                tmp_union = tmp_union | set(node_to_id_hashtag.values())
            if 'quote' in graphs_list:
                tmp_union = tmp_union | set(node_to_id_quote.values())
            if 'friend' in graphs_list:
                tmp_union = tmp_union | set(node_to_id_friend.values())
            if 'follow' in graphs_list:
                tmp_union = tmp_union | set(node_to_id_follow.values())
            if 'likes' in graphs_list:
                tmp_union = tmp_union | set(node_to_id_likes.values())
            tmp_union = tmp_union.intersection(set(id_to_true.keys())) - set(congress_users_id_to_true.keys())
            print(graphs_list, 'applicability:', round(100.0 * len(tmp_union) / total, 1))


        



