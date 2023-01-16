import pandas as pd
import copy
import json

def get_uid(inp):
    if inp in screen_name_to_id:
        return screen_name_to_id[inp]
    else:
        return 'MISSING'


with open("../config/common_data_paths.json", "r") as jsonfile:
    path_dict = json.load(jsonfile)

PROFILE_CLASSIFIER_USERS_PATH = path_dict["PROFILE_CLASSIFIER_USERS_PATH"]
SCREEN_NAME_TO_ID_PATH = path_dict["SCREEN_NAME_TO_ID_PATH"]
TRUE_LABELS_PATH = path_dict["TRUE_LABELS_PATH"]
OLD_USERS_PATH = path_dict["OLD_USERS_PATH"]

user_df = pd.read_json(OLD_USERS_PATH, lines=True)
screen_name_to_id = pd.read_csv(SCREEN_NAME_TO_ID_PATH,sep='\t', dtype={'user_id':str})
screen_name_to_id = pd.Series(screen_name_to_id.user_id.values,index=screen_name_to_id.screen_name).to_dict()

print('original weak label users', len(user_df))
user_df['UID'] = user_df['screen_name'].apply(get_uid)
user_df = user_df[user_df.UID != 'MISSING']
print('Weak label users matched to IDs:',len(user_df))

user_df['text'] = user_df['description']
filter_terms = ['conservative', 'gop', 'republican', 'trump', 'liberal', 'progressive', 'democrat', 'biden', 'progressive']
filtered_df = copy.copy(user_df)
filtered_df = filtered_df[~filtered_df.text.isna()]
filtered_df = filtered_df[filtered_df.text.str.lower().str.contains('|'.join(filter_terms))]

print('total public users with correct profile', len(filtered_df))

missing_uids = pd.read_json('../data/missing_uids_v2.json', dtype={'uid':str})
missing_uids = missing_uids.uid.tolist()

filtered_df = filtered_df[~filtered_df.UID.isin(missing_uids)]

print('total public users not missing with correct profile', len(filtered_df))

true_df = pd.read_json(TRUE_LABELS_PATH,orient='records',lines=True)
print(len(true_df))
true_df['UID'] = true_df['screen_name'].apply(get_uid)
true_df = true_df[true_df.UID != 'MISSING']
print(len(true_df))

true_id_list = true_df.UID.tolist()

filtered_df = filtered_df[~filtered_df.UID.isin(true_id_list)]

print('total public users not missing with correct profile, not true labeled', len(filtered_df))

filtered_df = filtered_df[['UID','text','screen_name']]

filtered_df.to_json('all_nontrue_users.jsonl',lines=True,orient='records')
print(filtered_df.head())
