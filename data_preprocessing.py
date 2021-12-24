import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import time
import numpy as np


def unique_list(l):
    ulist = []
    [ulist.append(x) for x in l if x not in ulist]
    return ulist


def load_data(num_data=0):
    # dataset_df = pd.read_pickle('Dataset/DS_train.pkl').sample(n=100000).reset_index(drop=True)
    if not num_data:
        dataset_df = pd.read_pickle('Dataset/DS_train.pkl')
    else:
        dataset_df = pd.read_pickle('Dataset/DS_train.pkl')[:num_data]

    msk = np.random.rand(len(dataset_df)) <= 0.8

    train_df = dataset_df[msk].reset_index(drop=True)
    valid_df = dataset_df[~msk].reset_index(drop=True)

    train_df = dataset_df
    train_output = train_df.emotion
    return train_df, train_output


def load_word_list(thr_saturation=0, thr_intensity=0):
    words_list = pd.read_pickle('hsv/hsv_list_spe.pkl')
    words_list_valid = words_list
    words_list_valid = words_list.loc[words_list.saturation >= thr_saturation]
    words_list_valid = words_list_valid.loc[words_list_valid.intensity >= thr_intensity]

    return words_list_valid


def load_word_counter(words_list_valid: pd.DataFrame()):
    words_counter = CountVectorizer()
    words_counter = words_counter.fit(words_list_valid.words)
    analyze = words_counter.build_analyzer()

    return words_counter, analyze


if __name__ == '__main__':
    # dataset_df = pd.read_pickle('Dataset/DS_train.pkl').sample(n=100000).reset_index(drop=True)
    # --- load data --- #
    train_df, train_output = load_data()

    # --- load word list --- #
    words_list_valid = load_word_list(thr_saturation=0, thr_intensity=8)

    # --- initiate the word counter --- #
    words_counter, analyze = load_word_counter(words_list_valid)

    output = pd.DataFrame()

    start= time.time()

    # --- initiate the parameters --- #
    results = []
    predict_emotion = []
    tweet_id = []
    total_duration = 0

    for index_tweets, tweet in enumerate(train_df.lemmas):

        # --- clean the repeated words --- #
        tweet = ' '.join(unique_list(tweet.split()))

        # --- start time counting --- #
        start = time.time()

        # --- initiate the temp sentence --- #
        train_sentence = pd.DataFrame()

        train_sentence = words_list_valid.loc[words_list_valid.words.isin(analyze(tweet)), 'anger':'trust']
        train_saturation = words_list_valid.loc[words_list_valid.words.isin(analyze(tweet)), 'saturation']
        train_sentence = train_sentence.mul(train_saturation, axis=0)

        score = train_sentence.sum(axis=0)

        # print(score)

        if len(score.to_list()):
            predict_emotion.append(train_sentence.sum(axis=0).idxmax())
            results.append(score.to_list())
        else:
            predict_emotion.append('joy')
            results.append([-1, -1, -1, -1, -1, -1, -1, -1])
        tweet_id.append(train_df.id[index_tweets])

        end = time.time()
        duration = end - start
        total_duration += duration
        progress = ((index_tweets + 1) / len(train_df)) * 100

        et = total_duration * 100 / progress
        td = total_duration
        eta = et - td

        et = time.strftime('%H:%M:%S', time.gmtime(total_duration * 100 / progress))
        td = time.strftime('%H:%M:%S', time.gmtime(total_duration))
        eta = time.strftime('%H:%M:%S', time.gmtime(eta))

        print(
            f'\r{progress:.2f}%, || PT: {td}, ET: {et}, ETA: {eta}', end="")
    
    output = pd.DataFrame({'id': tweet_id, 'predict_emotion': predict_emotion, 'score': results})
    output['output'] = train_output

    output.to_csv('result.csv')
    output.to_pickle('result.pkl')

    pass
