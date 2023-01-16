import pandas as pd
import pickle
import copy
import random
random.seed(42)
import numpy as np
np.random.seed(42)
import json

with open("config/common_data_paths.json", "r") as jsonfile:
    path_dict = json.load(jsonfile)
SCREEN_NAME_TO_ID_PATH = path_dict["SCREEN_NAME_TO_ID_PATH"]
TRUE_LABELS_PATH = path_dict["TRUE_LABELS_PATH"]


with open(../config/main_experiment.json, "r") as jsonfile:
    config_dict = json.load(jsonfile)
GRAPH_FOLDER = config_dict['GRAPH_FOLDER']

with open(f'{GRAPH_FOLDER}/test_ids_union.pkl', 'rb') as f:
        test_id_save_union = pickle.load(f)

screen_name_to_id = pd.read_csv(SCREEN_NAME_TO_ID_PATH,sep='\t', dtype={'user_id':str})
screen_name_to_id = pd.Series(screen_name_to_id.user_id.values,index=screen_name_to_id.screen_name).to_dict()
def get_id(inp):
    if inp in screen_name_to_id:
        return screen_name_to_id[inp]
    else:
        return 'NA'

true_df = pd.read_json(TRUE_LABELS_PATH,orient='records',lines=True)
true_df['UID'] = true_df.screen_name.apply(get_id)
true_df = true_df[true_df.UID != 'NA']

true_df['text'] = true_df['description']
true_df['label'] = true_df['true_label']

true_df = true_df[['UID','text','label']]

print(len(true_df))

test_df = true_df[true_df.UID.isin(test_id_save_union)]
print(test_df.head())
print(len(test_df))
test_df.to_json('test.jsonl',lines=True, orient='records')


train_df = true_df[~true_df.UID.isin(test_id_save_union)]
train_df = train_df.sample(frac=1.0)
valid_df = copy.copy(train_df[:100])
train_df = copy.copy(train_df[100:])
print(train_df.head())
print(len(train_df))
train_df.to_json('train.jsonl',lines=True, orient='records')
valid_df.to_json('val.jsonl',lines=True, orient='records')


print(set(train_df.text.tolist()).intersection(valid_df.text.tolist()))
print(set(train_df.text.tolist()).intersection(test_df.text.tolist()))