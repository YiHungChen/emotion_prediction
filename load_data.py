"""
This file is for the loading of the emotion data from the kaggle
"""


import json
import pandas as pd
import csv
import numpy as np
from tqdm import tqdm



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

rows = []

with open('dm2021-lab2-hw2/data_identification.csv') as csvfile:
    next(csvfile)
    for row in csv.reader(csvfile):
        rows.append(row)


tweets_df = tweets_df.sort_values("id")
rows.sort(key = lambda s: s[0])
rows_arr = np.array(rows)
tweets_df['identification'] = rows_arr[:, 1]

print(len(tweets_df))


for row in rows:
    tweets_df.loc[tweets_df.id == row[0], 'identification'] = row[1]
    pass