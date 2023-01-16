# This produces output for visualization, and runs a random forest to check MSE, MAE, and correlation when doing supervised DW-Nominate prediction
import sys
import copy

import json
import pandas as pd
pd.set_option('mode.chained_assignment', None) # turning off this warning, care
import numpy as np
np.random.seed(42)
import random
random.seed(42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import pickle
import os
import copy

CONFIG_PATH = sys.argv[1]

with open(CONFIG_PATH, "r") as jsonfile:
    config_dict = json.load(jsonfile)

with open("config/common_data_paths.json", "r") as jsonfile:
    path_dict = json.load(jsonfile)
PROFILE_CLASSIFIER_USERS_PATH = path_dict["PROFILE_CLASSIFIER_USERS_PATH"]
SCREEN_NAME_TO_ID_PATH = path_dict["SCREEN_NAME_TO_ID_PATH"]
TRUE_LABELS_PATH = path_dict["TRUE_LABELS_PATH"]
POLITICIANS_FOLDER = path_dict['POLITICIANS_FOLDER'] # path to politicians data folder

SEMI = config_dict["SEMI"]
TWO_LAYER = config_dict["TWO_LAYER"]
AUTOGLUON = False
INCLUDE_POLITICIANS = config_dict["INCLUDE_POLITICIANS"]
if "GAT" in config_dict:
    USE_GAT = config_dict["GAT"]

print('SEMI:', SEMI)


results_dict = {}
graphs_to_use_list = config_dict['GRAPHS_TO_USE_LIST']

for list_index, graphs_to_use in enumerate(graphs_to_use_list):
    results_dict[list_index] = {}
    test_acc_list_normal_normal = []
    test_acc_list_normal_pol = []
    test_acc_list_pol_normal = []
    test_acc_list_pol_pol = []
    test_acc_list_normalpol_normal = []
    test_acc_list_normalpol_pol = []
    autogluon_acc_list_normalpol_pol = []

    for repeat_num in range(config_dict["NUM_SPLITS"]):
        print('repeat number:',repeat_num)

        BASE_DATA_FOLDER = config_dict['GRAPH_FOLDER'] +'/' + str(repeat_num)
        if SEMI:
            if TWO_LAYER:
                GRAPH_EMBEDDING_FOLDER = BASE_DATA_FOLDER + f'/graph_embeddings_semi_twolayer'
                PROCESSED_EMBEDDINGS_FOLDER = BASE_DATA_FOLDER + f'/postprocessed_semi_twolayer' 
            else:
                GRAPH_EMBEDDING_FOLDER = BASE_DATA_FOLDER + f'/graph_embeddings_semi'
                PROCESSED_EMBEDDINGS_FOLDER = BASE_DATA_FOLDER + f'/postprocessed_semi' 
        else:
            if TWO_LAYER:
                GRAPH_EMBEDDING_FOLDER = BASE_DATA_FOLDER + f'/graph_embeddings_twolayer'
                PROCESSED_EMBEDDINGS_FOLDER = BASE_DATA_FOLDER + f'/postprocessed_twolayer' 
            else:
                GRAPH_EMBEDDING_FOLDER = BASE_DATA_FOLDER + f'/graph_embeddings' 
                PROCESSED_EMBEDDINGS_FOLDER = BASE_DATA_FOLDER + f'/postprocessed'
        if USE_GAT:
            GRAPH_EMBEDDING_FOLDER = BASE_DATA_FOLDER + f'/graph_embeddings_GAT'
            PROCESSED_EMBEDDINGS_FOLDER = BASE_DATA_FOLDER + f'/postprocessed_GAT'


        if not os.path.exists(PROCESSED_EMBEDDINGS_FOLDER):
            os.makedirs(PROCESSED_EMBEDDINGS_FOLDER)

        classifier_label_df = pd.read_json(PROFILE_CLASSIFIER_USERS_PATH, lines=True, orient='records')

        screen_name_to_id = pd.read_csv(SCREEN_NAME_TO_ID_PATH,sep='\t', dtype={'user_id':str})
        screen_name_to_id = pd.Series(screen_name_to_id.user_id.values,index=screen_name_to_id.screen_name).to_dict()


        true_df = pd.read_json(TRUE_LABELS_PATH,orient='records',lines=True)
        id_to_true = pd.Series(true_df.true_label.values,index=true_df.screen_name).to_dict()
        id_to_true_tmp = {str(key) : value for key, value in id_to_true.items()}
        id_to_true = {}
        for screen_name, value in id_to_true_tmp.items():
            if screen_name in screen_name_to_id:
                id_to_true[screen_name_to_id[screen_name]] = value
            else:
                continue


        id_to_weak = pd.Series(classifier_label_df.classifier_label.values,index=classifier_label_df.screen_name).to_dict()
        id_to_weak_tmp = {str(key) : value for key, value in id_to_weak.items()}
        id_to_weak = {}
        found_name_to_id_counter = 0
        for screen_name, value in id_to_weak_tmp.items():
            if screen_name in screen_name_to_id:
                id_to_weak[screen_name_to_id[screen_name]] = value
            else:
                continue


        if INCLUDE_POLITICIANS:
            congress_users_df = pd.read_json(f'{POLITICIANS_FOLDER}/congress_dataset_V2/congress_users.json', lines=True, orient='records')
            with open(f'{POLITICIANS_FOLDER}/congress_dataset_V2/congress_users.json', 'r') as f:
                lines = f.readlines()

            screen_name_to_id_congress_users = {}
            for line in lines:
                tmp_loaded = json.loads(line)
                screen_name_to_id_congress_users[tmp_loaded['json']['screen_name']] = tmp_loaded['json']['id_str']

            classifier_label_df_congress_users = pd.read_json(f'data/politicians_users_with_classifierlabel.jsonl', lines=True, orient='records')
            classifier_label_df_congress_users.true_label = classifier_label_df_congress_users.true_label.map({'Democratic':1,'Republican':0})
            true_label_df_congress_users = classifier_label_df_congress_users[~classifier_label_df_congress_users.true_label.isna()]
            screenname_to_true_congress_users = pd.Series(true_label_df_congress_users.true_label.values,index=true_label_df_congress_users.screen_name).to_dict()
            screenname_to_true_congress_users = {str(key) : value for key, value in screenname_to_true_congress_users.items()}
            congress_users_id_to_true = {}
            for key, value in screenname_to_true_congress_users.items():
                congress_users_id_to_true[screen_name_to_id_congress_users[key]] = value
            id_to_true.update(congress_users_id_to_true)





        classifier_stats = pd.DataFrame()
        imbalance_ratio = []

        
        key_list = set()

        try:
            if 'hashtag' in graphs_to_use:
                with open(f'{GRAPH_EMBEDDING_FOLDER}/hashtag.pkl', 'rb') as f:
                    save_dict_hashtag = pickle.load(f)
                embeddings_by_name_hashtag = save_dict_hashtag['embeddings_by_name']
            if 'retweet' in graphs_to_use:
                with open(f'{GRAPH_EMBEDDING_FOLDER}/retweet.pkl', 'rb') as f:
                    save_dict_retweet = pickle.load(f)
                embeddings_by_name_retweet = save_dict_retweet['embeddings_by_name']
            if 'mention' in graphs_to_use:
                with open(f'{GRAPH_EMBEDDING_FOLDER}/mention.pkl', 'rb') as f:
                    save_dict_mention = pickle.load(f)
                embeddings_by_name_mention = save_dict_mention['embeddings_by_name']
            if 'quote' in graphs_to_use:
                with open(f'{GRAPH_EMBEDDING_FOLDER}/quote.pkl', 'rb') as f:
                    save_dict_quote = pickle.load(f)
                embeddings_by_name_quote = save_dict_quote['embeddings_by_name']
            if 'follow' in graphs_to_use:
                with open(f'{GRAPH_EMBEDDING_FOLDER}/follow.pkl', 'rb') as f:
                    save_dict_follow = pickle.load(f)
                embeddings_by_name_follow = save_dict_follow['embeddings_by_name']
            if 'friend' in graphs_to_use:
                with open(f'{GRAPH_EMBEDDING_FOLDER}/friend.pkl', 'rb') as f:
                    save_dict_friend = pickle.load(f)
                embeddings_by_name_friend = save_dict_friend['embeddings_by_name']
            if 'likes' in graphs_to_use:
                with open(f'{GRAPH_EMBEDDING_FOLDER}/likes.pkl', 'rb') as f:
                    save_dict_likes = pickle.load(f)
                embeddings_by_name_likes = save_dict_likes['embeddings_by_name']
            
        except:
            print(f'failed (no data?)')
            continue
        
        
        

        if 'text' in graphs_to_use:
            if config_dict["SEPARATE_TEXT_EMBEDDING_FILES"]:
                with open(f'{config_dict["PATH_TO_TEXT_MODEL1"]}-{repeat_num}-{config_dict["PATH_TO_TEXT_MODEL2"]}', 'r') as f:
                    save_dict_text = json.load(f)
            else:
                with open(f'{config_dict["PATH_TO_TEXT_MODEL"]}', 'r') as f:
                    save_dict_text = json.load(f)
            len_text_embeddings = len(save_dict_text[list(save_dict_text.keys())[0]])
            embeddings_by_name_text = {str(key) : np.array(value) for key, value in save_dict_text.items()}
            if len(graphs_to_use) == 1:
                with open(f"{config_dict['GRAPH_FOLDER']}/test_ids.pkl", 'rb') as f:
                    test_ids_dict = pickle.load(f)
                    text_test_ids_list = test_ids_dict[repeat_num]
                    key_list = key_list | set(embeddings_by_name_text.keys())


        if 'mention' in graphs_to_use:
            key_list = key_list | set(embeddings_by_name_mention.keys())
        if 'retweet' in graphs_to_use:
            key_list = key_list | set(embeddings_by_name_retweet.keys())
        if 'hashtag' in graphs_to_use:
            key_list = key_list | set(embeddings_by_name_hashtag.keys())
        if 'quote' in graphs_to_use:
            key_list = key_list | set(embeddings_by_name_quote.keys())
        if 'follow' in graphs_to_use:
            key_list = key_list | set(embeddings_by_name_follow.keys())
        if 'friend' in graphs_to_use:
            key_list = key_list | set(embeddings_by_name_friend.keys())
        if 'likes' in graphs_to_use:
            key_list = key_list | set(embeddings_by_name_likes.keys())



        print(f'users considered in {graphs_to_use} run {repeat_num}:', len(key_list))

        if INCLUDE_POLITICIANS:
            key_list = key_list.intersection(set(screen_name_to_id.values()) | set(screen_name_to_id_congress_users.values()))
        else:
            key_list = key_list.intersection(set(screen_name_to_id.values()))



        key_list = list(key_list)



        embeddings_combined_train = []
        embeddings_combined_val = []
        embeddings_combined_test = []
        embeddings_combined_politicians = []
        valid_key_list_train = []
        valid_key_list_val = []
        valid_key_list_test = []
        valid_key_list_unsupervised = []
        valid_key_list_politicians = []
        embeddings_combined_unsupervised = []
        embeddings_all_four = []

        bad_counter = 0
        random.shuffle(key_list)
        for key in key_list:
            train_flag = 0
            val_flag = 0
            test_flag = 0
            politician_flag = 0

            tmp_retweet = 1000*np.zeros((100))
            
            if 'retweet' in graphs_to_use:
                if key in embeddings_by_name_retweet:
                    tmp_retweet = embeddings_by_name_retweet[key]
                    if save_dict_retweet['handle_to_node'][key] in save_dict_retweet['train_nodes']:
                        train_flag = 1
                    if save_dict_retweet['handle_to_node'][key] in save_dict_retweet['val_nodes']:
                        val_flag = 1
                    if save_dict_retweet['handle_to_node'][key] in save_dict_retweet['test_nodes']:
                        test_flag = 1
                    if INCLUDE_POLITICIANS:
                        if save_dict_retweet['handle_to_node'][key] in save_dict_retweet['politician_nodes']:
                            politician_flag = 1
            
            
            tmp_hashtag = 1000*np.zeros((100))
            
            if 'hashtag' in graphs_to_use:
                if key in embeddings_by_name_hashtag:
                    tmp_hashtag = embeddings_by_name_hashtag[key]
                    if save_dict_hashtag['handle_to_node'][key] in save_dict_hashtag['train_nodes']:
                        train_flag = 1
                    if save_dict_hashtag['handle_to_node'][key] in save_dict_hashtag['val_nodes']:
                        val_flag = 1
                    if save_dict_hashtag['handle_to_node'][key] in save_dict_hashtag['test_nodes']:
                        test_flag = 1
                    if INCLUDE_POLITICIANS:
                        if save_dict_hashtag['handle_to_node'][key] in save_dict_hashtag['politician_nodes']:
                            politician_flag = 1
            
            tmp_quote = 1000*np.zeros((100))
            
            if 'quote' in graphs_to_use:
                if key in embeddings_by_name_quote:
                    tmp_quote = embeddings_by_name_quote[key]
                    if save_dict_quote['handle_to_node'][key] in save_dict_quote['train_nodes']:
                        train_flag = 1
                    if save_dict_quote['handle_to_node'][key] in save_dict_quote['val_nodes']:
                        val_flag = 1
                    if save_dict_quote['handle_to_node'][key] in save_dict_quote['test_nodes']:
                        test_flag = 1
                    if INCLUDE_POLITICIANS:
                        if save_dict_quote['handle_to_node'][key] in save_dict_quote['politician_nodes']:
                            politician_flag = 1
            

            tmp_mention = 1000*np.zeros((100))
            
            if 'mention' in graphs_to_use:
                if key in embeddings_by_name_mention:
                    tmp_mention = embeddings_by_name_mention[key]
                    if save_dict_mention['handle_to_node'][key] in save_dict_mention['train_nodes']:
                        train_flag = 1
                    if save_dict_mention['handle_to_node'][key] in save_dict_mention['val_nodes']:
                        val_flag = 1
                    if save_dict_mention['handle_to_node'][key] in save_dict_mention['test_nodes']:
                        test_flag = 1
                    if INCLUDE_POLITICIANS:
                        if save_dict_mention['handle_to_node'][key] in save_dict_mention['politician_nodes']:
                            politician_flag = 1
            

            tmp_follow = 1000*np.zeros((100))
            
            if 'follow' in graphs_to_use:
                if key in embeddings_by_name_follow:
                    tmp_follow = embeddings_by_name_follow[key]
                    if save_dict_follow['handle_to_node'][key] in save_dict_follow['train_nodes']:
                        train_flag = 1
                    if save_dict_follow['handle_to_node'][key] in save_dict_follow['val_nodes']:
                        val_flag = 1
                    if save_dict_follow['handle_to_node'][key] in save_dict_follow['test_nodes']:
                        test_flag = 1
                    if INCLUDE_POLITICIANS:
                        if save_dict_follow['handle_to_node'][key] in save_dict_follow['politician_nodes']:
                            politician_flag = 1

            tmp_friend = 1000*np.zeros((100))
            
            if 'friend' in graphs_to_use:
                if key in embeddings_by_name_friend:
                    tmp_friend = embeddings_by_name_friend[key]
                    if save_dict_friend['handle_to_node'][key] in save_dict_friend['train_nodes']:
                        train_flag = 1
                    if save_dict_friend['handle_to_node'][key] in save_dict_friend['val_nodes']:
                        val_flag = 1
                    if save_dict_friend['handle_to_node'][key] in save_dict_friend['test_nodes']:
                        test_flag = 1
                    if INCLUDE_POLITICIANS:
                        if save_dict_friend['handle_to_node'][key] in save_dict_friend['politician_nodes']:
                            politician_flag = 1

            tmp_likes = 1000*np.zeros((100))
            
            if 'likes' in graphs_to_use:
                if key in embeddings_by_name_likes:
                    tmp_likes = embeddings_by_name_likes[key]
                    if save_dict_likes['handle_to_node'][key] in save_dict_likes['train_nodes']:
                        train_flag = 1
                    if save_dict_likes['handle_to_node'][key] in save_dict_likes['val_nodes']:
                        val_flag = 1
                    if save_dict_likes['handle_to_node'][key] in save_dict_likes['test_nodes']:
                        test_flag = 1
                    #if save_dict_likes['handle_to_node'][key] in save_dict_likes['politician_nodes']:
                    #    politician_flag = 1

            if 'text' in graphs_to_use:
                tmp_text = 1000*np.zeros((len_text_embeddings))
                if str(key) in embeddings_by_name_text:
                    tmp_text = embeddings_by_name_text[key]
                    if key in id_to_weak and key not in id_to_true:
                        train_flag = 1
                    if key in id_to_true and key not in text_test_ids_list:
                        val_flag = 1
                    if key in text_test_ids_list:
                        test_flag = 1


            if 'mention' == graphs_to_use[0]:
                tmp_embedding = np.concatenate([tmp_mention])
            elif 'retweet' == graphs_to_use[0]:
                tmp_embedding = np.concatenate([tmp_retweet])
            elif 'hashtag' == graphs_to_use[0]:
                tmp_embedding = np.concatenate([tmp_hashtag])
            elif 'quote' == graphs_to_use[0]:
                tmp_embedding = np.concatenate([tmp_quote])
            elif 'follow' == graphs_to_use[0]:
                tmp_embedding = np.concatenate([tmp_follow])
            elif 'friend' == graphs_to_use[0]:
                tmp_embedding = np.concatenate([tmp_friend])
            elif 'likes' == graphs_to_use[0]:
                tmp_embedding = np.concatenate([tmp_likes])
            elif 'text' == graphs_to_use[0]:
                tmp_embedding = np.concatenate([tmp_text])
            if len(graphs_to_use) > 1:
                if 'mention' in graphs_to_use[1:]:
                    tmp_embedding = np.concatenate([tmp_embedding,tmp_mention])
                if 'retweet' in graphs_to_use[1:]:
                    tmp_embedding = np.concatenate([tmp_embedding,tmp_retweet])
                if 'hashtag' in graphs_to_use[1:]:
                    tmp_embedding = np.concatenate([tmp_embedding,tmp_hashtag])
                if 'quote' in graphs_to_use[1:]:
                    tmp_embedding = np.concatenate([tmp_embedding,tmp_quote])
                if 'follow' in graphs_to_use[1:]:
                    tmp_embedding = np.concatenate([tmp_embedding,tmp_follow])
                if 'friend' in graphs_to_use[1:]:
                    tmp_embedding = np.concatenate([tmp_embedding,tmp_friend])
                if 'likes' in graphs_to_use[1:]:
                    tmp_embedding = np.concatenate([tmp_embedding,tmp_likes])
                if 'text' in graphs_to_use[1:]:
                    tmp_embedding = np.concatenate([tmp_embedding,tmp_text])


            if (train_flag == 1 or val_flag == 1) and test_flag == 1:
                print('ERROR: in multiple parts of split')
                print(train_flag,val_flag,test_flag)
                bad_counter += 1
                if ((key in id_to_true) or (key in id_to_weak)):
                    embeddings_combined_unsupervised.append(tmp_embedding)
                    valid_key_list_unsupervised.append(key)
            elif train_flag == 1 and politician_flag == 0:
                embeddings_combined_train.append(tmp_embedding)
                valid_key_list_train.append(key)
            elif val_flag == 1 and politician_flag == 0:
                embeddings_combined_val.append(tmp_embedding)
                valid_key_list_val.append(key)
            elif test_flag == 1 and politician_flag == 0:
                embeddings_combined_test.append(tmp_embedding)
                valid_key_list_test.append(key)
            else:
                if ((key in id_to_true) or (key in id_to_weak)):
                    embeddings_combined_unsupervised.append(tmp_embedding)
                    valid_key_list_unsupervised.append(key)
            if politician_flag == 1:
                    embeddings_combined_politicians.append(tmp_embedding)
                    valid_key_list_politicians.append(key)

        


        if bad_counter > 0:
            print('CAUTION CAUTION CAUTION leaking:', bad_counter)
        if repeat_num == 0:
            print('train/val/test sizes', len(embeddings_combined_train),len(embeddings_combined_val),len(embeddings_combined_test))

        
        
        # NORMAL -> NORMAL
        
        train_labels = [id_to_weak[uid] for uid in valid_key_list_train]
        counter_true = 0
        for uid in valid_key_list_train:
            if uid in id_to_true:
                counter_true += 1

        train_labels = pd.Series(train_labels).map({'liberal' : 0, 'conservative' : 1, 0:1, 1:0}).to_list()
        test_labels = [id_to_true[uid] for uid in valid_key_list_test]

        X_train = np.array(embeddings_combined_train)
        y_train = train_labels
        X_test = np.array(embeddings_combined_test)
        y_test = 1 - pd.Series(test_labels) 

        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)

        y_pred_test = rf.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred_test)
        test_acc_list_normal_normal.append(test_acc)

        print(test_acc)

        if INCLUDE_POLITICIANS:
            # NORMAL -> POLITICIAN
            train_labels = [id_to_weak[uid] for uid in valid_key_list_train]
            counter_true = 0
            for uid in valid_key_list_train:
                if uid in id_to_true:
                    counter_true += 1

            train_labels = pd.Series(train_labels).map({'liberal' : 0, 'conservative' : 1, 0:1, 1:0}).to_list()
            test_labels = [id_to_true[uid] for uid in valid_key_list_politicians]

            X_train = np.array(embeddings_combined_train)
            y_train = train_labels
            X_test = np.array(embeddings_combined_politicians)
            y_test = 1 - pd.Series(test_labels) 

            rf = RandomForestClassifier()
            rf.fit(X_train, y_train)

            y_pred_test = rf.predict(X_test)
            test_acc = accuracy_score(y_test, y_pred_test)
            test_acc_list_normal_pol.append(test_acc)






            # POLITICIAN -> NORMAL

            train_labels = [id_to_true[uid] for uid in valid_key_list_politicians]
            counter_true = 0
            for uid in valid_key_list_train:
                if uid in id_to_true:
                    counter_true += 1

            train_labels = pd.Series(train_labels).map({'liberal' : 0, 'conservative' : 1, 0:1, 1:0}).to_list()
            test_labels = [id_to_true[uid] for uid in valid_key_list_test]

            X_train = np.array(embeddings_combined_politicians)
            y_train = train_labels
            X_test = np.array(embeddings_combined_test)
            y_test = 1 - pd.Series(test_labels) 

            rf = RandomForestClassifier()
            rf.fit(X_train, y_train)

            y_pred_test = rf.predict(X_test)
            test_acc = accuracy_score(y_test, y_pred_test)
            test_acc_list_pol_normal.append(test_acc)

            

            # POLITICIAN -> POLITICIAN

            train_labels = [id_to_true[uid] for uid in valid_key_list_politicians[:int(len(valid_key_list_politicians) * .75)]]
            counter_true = 0
            for uid in valid_key_list_train:
                if uid in id_to_true:
                    counter_true += 1

            train_labels = pd.Series(train_labels).map({'liberal' : 0, 'conservative' : 1, 0:1, 1:0}).to_list()
            test_labels = [id_to_true[uid] for uid in valid_key_list_politicians[int(len(valid_key_list_politicians) * .75):]]

            X_train = np.array(embeddings_combined_politicians[:int(len(valid_key_list_politicians) * .75)])
            y_train = train_labels
            X_test = np.array(embeddings_combined_politicians[int(len(valid_key_list_politicians) * .75):])
            y_test = 1 - pd.Series(test_labels) 

            rf = RandomForestClassifier()
            rf.fit(X_train, y_train)


            y_pred_test = rf.predict(X_test)
            test_acc = accuracy_score(y_test, y_pred_test)
            test_acc_list_pol_pol.append(test_acc)



            

            # NORMAL + POLITICIAN -> NORMAL

            train_labels = [id_to_true[uid] for uid in valid_key_list_politicians] + [id_to_weak[uid] for uid in valid_key_list_train] 
            counter_true = 0
            for uid in valid_key_list_train:
                if uid in id_to_true:
                    counter_true += 1

            train_labels = pd.Series(train_labels).map({'liberal' : 0, 'conservative' : 1, 0:1, 1:0}).to_list()
            test_labels = [id_to_true[uid] for uid in valid_key_list_test]

            X_train = np.array(embeddings_combined_politicians + embeddings_combined_train)
            y_train = train_labels
            X_test = np.array(embeddings_combined_test)
            y_test = 1 - pd.Series(test_labels) 

            rf = RandomForestClassifier()
            rf.fit(X_train, y_train)


            y_pred_test = rf.predict(X_test)
            test_acc = accuracy_score(y_test, y_pred_test)
            test_acc_list_normalpol_normal.append(test_acc)



            
            # NORMAL + POLITICIAN -> POLITICIAN

            train_labels = [id_to_true[uid] for uid in valid_key_list_politicians[:int(len(valid_key_list_politicians) * .75)]] + [id_to_weak[uid] for uid in valid_key_list_train]
            counter_true = 0
            for uid in valid_key_list_train:
                if uid in id_to_true:
                    counter_true += 1

            train_labels = pd.Series(train_labels).map({'liberal' : 0, 'conservative' : 1, 0:1, 1:0}).to_list()
            test_labels = [id_to_true[uid] for uid in valid_key_list_politicians[int(len(valid_key_list_politicians) * .75):]]

            X_train = np.array(embeddings_combined_politicians[:int(len(valid_key_list_politicians) * .75)] + embeddings_combined_train)
            y_train = train_labels
            X_test = np.array(embeddings_combined_politicians[int(len(valid_key_list_politicians) * .75):])
            y_test = 1 - pd.Series(test_labels) 

            rf = RandomForestClassifier()
            rf.fit(X_train, y_train)

            y_pred_test = rf.predict(X_test)
            test_acc = accuracy_score(y_test, y_pred_test)
            test_acc_list_normalpol_pol.append(test_acc)
        

    results_dict[list_index]['normal_normal'] = test_acc_list_normal_normal
    if INCLUDE_POLITICIANS:
        results_dict[list_index]['normal_pol'] = test_acc_list_normal_pol
        results_dict[list_index]['pol_normal'] = test_acc_list_pol_normal
        results_dict[list_index]['pol_pol'] = test_acc_list_pol_pol
        results_dict[list_index]['normalpol_normal'] = test_acc_list_normalpol_normal
        results_dict[list_index]['normalpol_pol'] = test_acc_list_normalpol_pol
        results_dict[list_index]['autogluon_normalpol_pol'] = autogluon_acc_list_normalpol_pol

results_df = pd.DataFrame()

for key in results_dict.keys():
    print(graphs_to_use_list[key])
    results_df = pd.concat([results_df, pd.DataFrame({'data_types':graphs_to_use_list[key], 'train_data':'normal', 'test_data':'normal', 'mean':round(100*np.mean(results_dict[key]['normal_normal']),1), 'std':round(100*np.std(results_dict[key]['normal_normal']),1)})])
    print('AVERAGE TEST ACC, normal -> normal', round(100*np.mean(results_dict[key]['normal_normal']),1), '+-', round(100*np.std(results_dict[key]['normal_normal']),1))
    if INCLUDE_POLITICIANS:
        results_df = pd.concat([results_df, pd.DataFrame({'data_types':graphs_to_use_list[key], 'train_data':'normal', 'test_data':'politician', 'mean':round(100*np.mean(results_dict[key]['normal_normal']),1), 'std':round(100*np.std(results_dict[key]['normal_normal']),1)})])
        results_df = pd.concat([results_df, pd.DataFrame({'data_types':graphs_to_use_list[key], 'train_data':'politician', 'test_data':'normal', 'mean':round(100*np.mean(results_dict[key]['normal_normal']),1), 'std':round(100*np.std(results_dict[key]['normal_normal']),1)})])
        results_df = pd.concat([results_df, pd.DataFrame({'data_types':graphs_to_use_list[key], 'train_data':'politician', 'test_data':'politician', 'mean':round(100*np.mean(results_dict[key]['normal_normal']),1), 'std':round(100*np.std(results_dict[key]['normal_normal']),1)})])
        results_df = pd.concat([results_df, pd.DataFrame({'data_types':graphs_to_use_list[key], 'train_data':'normalpol', 'test_data':'normal', 'mean':round(100*np.mean(results_dict[key]['normal_normal']),1), 'std':round(100*np.std(results_dict[key]['normal_normal']),1)})])
        results_df = pd.concat([results_df, pd.DataFrame({'data_types':graphs_to_use_list[key], 'train_data':'normalpol', 'test_data':'politician', 'mean':round(100*np.mean(results_dict[key]['normal_normal']),1), 'std':round(100*np.std(results_dict[key]['normal_normal']),1)})])
        print('AVERAGE TEST ACC, normal -> politician', round(100*np.mean(results_dict[key]['normal_pol']),1), '+-', round(100*np.std(results_dict[key]['normal_pol']),1))
        print('AVERAGE TEST ACC, politician -> normal', round(100*np.mean(results_dict[key]['pol_normal']),1), '+-', round(100*np.std(results_dict[key]['pol_normal']),1))
        print('AVERAGE TEST ACC, politician -> politician', round(100*np.mean(results_dict[key]['pol_pol']),1), '+-', round(100*np.std(results_dict[key]['pol_pol']),1))
        print('AVERAGE TEST ACC, normal + politician -> normal', round(100*np.mean(results_dict[key]['normalpol_normal']),1), '+-', round(100*np.std(results_dict[key]['normalpol_normal']),1))
        print('AVERAGE TEST ACC, normal + politician -> politician', round(100*np.mean(results_dict[key]['normalpol_pol']),1), '+-', round(100*np.std(results_dict[key]['normalpol_pol']),1))
    partial_path_to_save = CONFIG_PATH.split('/')[-1].replace('json','csv')
    results_df.to_csv(f'results/{partial_path_to_save}', index=False)

