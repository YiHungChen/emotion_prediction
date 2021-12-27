"""
This file is for the loading of the emotion data from the kaggle
"""

import json
import pandas as pd
import re
from lemmatization import lemmas_words
from nltk.tokenize.treebank import TreebankWordDetokenizer
from tqdm import tqdm
from folder_path import folder_path

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
    x = re.sub("<LH>", "", x)
    x = re.sub('\'s', " is", x)
    x = re.sub('\'ve', " have", x)
    x = re.sub('n\'t', " not", x)
    x = re.sub('\'re', " are", x)
    x = re.sub('\'m', " am", x)
    x = re.sub('_', " ", x)
    x = re.sub(r'\'ll', "will", x)
    x = re.sub(r'@\w+', "", x)
    x = re.sub(r'#', "", x)

    output.append(TreebankWordDetokenizer().detokenize(lemmas_words(x)))
    return output

value = tqdm(tweets_text)


# tweets_text = tweets_text[:10]
# tweets_text = tweets_text.apply(lambda x: data_cleaning(x))
list(map(data_cleaning, value))
# tweets_text.map(data_cleaning())

tweets_df['id'] = tweets_id
tweets_df['text'] = tweets_text
tweets_df['lemmas'] = output

tweets_ident = pd.read_csv('dm2021-lab2-hw2/data_identification.csv')
tweets_ident = tweets_ident.rename(columns={"tweet_id": "id"})

tweets_df = pd.merge(tweets_df, tweets_ident, on=['id'])

tweets_df.to_pickle(f'{folder_path}Dataset/DS.pkl')
print(f'{folder_path}Dataset/DS.pkl is saved !!!')

tweets_emotion = pd.read_csv('dm2021-lab2-hw2/emotion.csv')
tweets_emotion = tweets_emotion.rename(columns={"tweet_id": "id"})

tweets_df = pd.merge(tweets_df, tweets_emotion, on=['id'])


tweets_df.to_pickle(f'{folder_path}Dataset/DS_train.pkl')
# tweets_df.to_csv('DS_train.csv')

print(f'{folder_path}Dataset/DS_train.pkl is saved !!!')
