"""
This file is for the loading of the emotion data from the kaggle
"""

import json
import pandas as pd
import re
import csv
import numpy as np
from lemmatization import lemmas_words

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

for index, x in enumerate(tweets_text):
    output = []
    x = x.replace("<LH>", "")
    x = x.replace('\'s', " is")
    x = x.replace('\'ve', " have")
    x = x.replace('n\'t', " not")
    x = x.replace('\'re', " are")
    x = x.replace('\'m', "am")

    x = re.sub("[0-9]", "", x)
    x = lemmas_words(x)
    print(f'\r{(index/len(tweets_text))*100 :.2f}%', end="")

    pass







tweets_df['id'] = tweets_id
tweets_df['text'] = tweets_text




tweets_ident = pd.read_csv('dm2021-lab2-hw2/data_identification.csv')
tweets_ident = tweets_ident.rename(columns={"tweet_id": "id"})

tweets_df = pd.merge(tweets_df, tweets_ident, on=['id'])

tweets_emotion = pd.read_csv('dm2021-lab2-hw2/emotion.csv')
tweets_emotion = tweets_emotion.rename(columns={"tweet_id": "id"})

tweets_df = pd.merge(tweets_df, tweets_emotion, on=['id'])


tweets_df.to_pickle('Dataset/DS_train.pkl')
tweets_df.to_csv('DS_train.csv')
print(len(tweets_df))
