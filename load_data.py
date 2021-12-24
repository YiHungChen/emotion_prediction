"""
This file is for the loading of the emotion data from the kaggle
"""

import json
import pandas as pd
import re
import csv
import numpy as np
from lemmatization import lemmas_words
import time
from nltk.tokenize.treebank import TreebankWordDetokenizer
from tqdm import tqdm
import multiprocessing

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

total_duration = 0
output = []


def data_cleaning(x):
    x = x.replace("<LH>", "")
    x = x.replace('\'s', " is")
    x = x.replace('\'ve', " have")
    x = x.replace('n\'t', " not")
    x = x.replace('\'re', " are")
    x = x.replace('\'m', " am")
    x = x.replace('\'ll', " will")
    x = x.replace('_', " ")

    x = re.sub('[0-9]', "", x)
    x = re.sub(r'@\w+', "", x)
    x = re.sub(r'#\w+', "", x)
    # x = re.sub("", "", x)
    # x = re.sub("", "", x)
    output.append(TreebankWordDetokenizer().detokenize(lemmas_words(x)))


    pass

# tweets_text = tweets_text.apply(lambda x: data_cleaning(x))
value = tqdm(tweets_text)

tweets_text = list(zip(*map(data_cleaning, value)))
# tweets_text.map(data_cleaning())





for index, x in enumerate(tweets_text):
    start = time.time()
    x = x.replace("<LH>", "")
    x = x.replace('\'s', " is")
    x = x.replace('\'ve', " have")
    x = x.replace('n\'t', " not")
    x = x.replace('\'re', " are")
    x = x.replace('\'m', " am")
    x = x.replace('\'ll', " will")
    x = x.replace('_', " ")

    x = re.sub("[0-9]", "", x)
    output.append(TreebankWordDetokenizer().detokenize(lemmas_words(x)))

    end = time.time()
    duration = end - start
    total_duration += duration
    progress = ((index + 1) / len(tweets_text)) * 100

    et = total_duration*100/progress
    td = total_duration
    eta = et-td

    et = time.strftime('%H:%M:%S', time.gmtime(total_duration*100/progress))
    td = time.strftime('%H:%M:%S', time.gmtime(total_duration))
    eta = time.strftime('%H:%M:%S', time.gmtime(eta))

    print(f'\r{progress:.2f}%, || PT: {td}, ET: {et}, ETA: {eta}', end="")
    pass
print("")


tweets_df['id'] = tweets_id
tweets_df['text'] = tweets_text
tweets_df['lemmas'] = output

tweets_ident = pd.read_csv('dm2021-lab2-hw2/data_identification.csv')
tweets_ident = tweets_ident.rename(columns={"tweet_id": "id"})

tweets_df = pd.merge(tweets_df, tweets_ident, on=['id'])

tweets_df.to_pickle('Dataset/DS.pkl')
print('Dataset/DS.pkl is saved !!!')

tweets_emotion = pd.read_csv('dm2021-lab2-hw2/emotion.csv')
tweets_emotion = tweets_emotion.rename(columns={"tweet_id": "id"})

tweets_df = pd.merge(tweets_df, tweets_emotion, on=['id'])


tweets_df.to_pickle('Dataset/DS_train.pkl')
tweets_df.to_csv('DS_train.csv')

print('Dataset/DS_train.pkl is saved !!!')
