"""
This file is for the loading of the emotion data from the kaggle
"""

import json
import pandas as pd
import csv
import numpy as np

tweets_df = pd.DataFrame()

with open('dm2021-lab2-hw2/tweets_DM.json') as jsonfile:
    tweets_id = []
    tweets_text = []
    tweets_emotion = []
    for tweet in jsonfile.readlines():
        dic = json.loads(tweet)
        tweets_id.append(dic['_source']['tweet']['tweet_id'])
        tweets_text.append(dic['_source']['tweet']['text'])

    pass

tweets_df['id'] = tweets_id
tweets_df['text'] = tweets_text
tweets_df['identification'] = np.nan

rows_ident = []
rows_emotion = []

with open('dm2021-lab2-hw2/data_identification.csv') as csv_file:
    next(csv_file)
    for row in csv.reader(csv_file):
        rows_ident.append(row)
        pass
    pass

with open('dm2021-lab2-hw2/emotion.csv') as emotion_file:
    next(emotion_file)
    for row in csv.reader(emotion_file):
        rows_emotion.append(row)
        pass
    pass

tweets_df = tweets_df.sort_values("id")
rows_ident.sort(key=lambda s: s[0])
rows_ident_arr = np.array(rows_ident)
tweets_df['identification'] = rows_ident_arr[:, 1]

id_list = tweets_df.id.to_list()
rows_emotion_arr = np.array(rows_emotion)
id_emotion = rows_emotion_arr[:, 0].tolist()

id_difference = list(set(id_list) - set(id_emotion))

tweets_df_reindex = tweets_df.set_index("id")
tweets_df_reindex = tweets_df_reindex.drop(id_difference)

rows_emotion.sort(key=lambda s: s[0])
tweets_df_reindex['emotion'] = rows_emotion_arr[:, 1]


tweets_df_reindex.to_pickle('DS.pkl')
print(len(tweets_df_reindex))
