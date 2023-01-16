import pandas as pd

politicians_df = pd.read_json('../data/politicians_users_with_classifierlabel.jsonl', lines=True, dtype={'uid':str})

actual_politicians = politicians_df[(politicians_df.true_label == 'Democratic') | (politicians_df.true_label == 'Republican')]

print('POLITICIANS')
print(len(actual_politicians))
print(actual_politicians.true_label.value_counts())
print(actual_politicians.classifier_label.value_counts())


filter_terms = ['conservative', 'gop', 'republican', 'trump', 'liberal', 'progressive', 'democrat', 'biden', 'progressive', 'progress']
profile_politicians = actual_politicians[~actual_politicians.text.isna()]
profile_politicians = profile_politicians[profile_politicians.text.str.lower().str.contains('|'.join(filter_terms))]



print('PROFILE POLITICIANS')
print(len(profile_politicians))
print(profile_politicians.true_label.value_counts())

print(profile_politicians.head())

profile_politicians = profile_politicians[['screen_name','text','true_label']]

profile_politicians.to_json('profile_politicians.jsonl',lines=True,orient='records')