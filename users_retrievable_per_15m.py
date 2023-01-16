import json
import numpy as np
import pandas as pd
import glob

with open('data/user_tweet_counts.json','r') as f:
    user_tweet_counts = json.load(f)

user_apicall_counts = {key : value // 200 + 1 for key, value in user_tweet_counts.items()}
apicall_counts_list = list(user_apicall_counts.values())

print('TWEETS')
calls_per_15m = 900
print(calls_per_15m / np.mean(apicall_counts_list), '+-',  np.sqrt(calls_per_15m*np.var(apicall_counts_list)))


with open("config/common_data_paths.json", "r") as jsonfile:
    path_dict = json.load(jsonfile)

FRIENDS_FOLDER = path_dict['FRIENDS_FOLDER'] # path to friends data folder
FOLLOWERS_FOLDER = path_dict['FOLLOWERS_FOLDER'] # path to followers data folder
LIKES_PATH = path_dict['LIKES_PATH'] # path to likes data file



def read_likes():
    df = pd.read_json(LIKES_PATH,lines=True,orient='records', dtype={'dest':str,'source':str,'dest_author_id':str})

    df = df.groupby(['source']).size().reset_index(name='count')

    total_relations = len(df)
    print('total likes', total_relations)



    print(len(df), len(list(df.source.unique())))

    return df



def read_friends():

    file_list = glob.glob(FRIENDS_FOLDER)

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
        len_list.append(len(tmp_follower))


    print('average friends per user:', np.mean(len_list))
    print('total friends for non-politicians', np.sum(len_list))
    print('max friends for non-politicians', np.max(len_list))

    return len_list

def read_followers():

    file_list = glob.glob(FOLLOWERS_FOLDER)

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
        len_list.append(len(tmp_follower))

    print('average followers per user:', np.mean(len_list))
    print('total followers for non-politicians', np.sum(len_list))
    print('max followers for non-politicians', np.max(len_list))

    return len_list


friends_list = read_friends()
apicall_list = [v // 5000 + 1 for v in friends_list]
calls_per_15m = 15
print('FRIENDS')
print(calls_per_15m / np.mean(apicall_list), '+-',  np.sqrt(calls_per_15m*np.var(apicall_list)))

followers_list = read_followers()
apicall_list = [v // 5000 + 1 for v in followers_list]
calls_per_15m = 15
print('FOLLOWERS')
print(calls_per_15m / np.mean(apicall_list), '+-',  np.sqrt(calls_per_15m*np.var(apicall_list)))

likes_df = read_likes()
likes_list = likes_df["count"].tolist()
apicall_list = [v // 100 + 1 for v in likes_list]
calls_per_15m = 15
print('LIKES')
print(calls_per_15m / np.mean(apicall_list), '+-',  np.sqrt(calls_per_15m*np.var(apicall_list)))
