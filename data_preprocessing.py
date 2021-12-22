import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import time
import numpy as np


def unique_list(l):
    ulist = []
    [ulist.append(x) for x in l if x not in ulist]
    return ulist



if __name__ == '__main__':
    dataset_df = pd.read_pickle('Dataset/DS_train.pkl').sample(n=10000).reset_index(drop=True)
    msk = np.random.rand(len(dataset_df)) <= 0.8

    train_df = dataset_df[msk].reset_index(drop=True)
    valid_df = dataset_df[~msk].reset_index(drop=True)

    train_output = train_df.emotion

    words_list = pd.read_pickle('Dataset/words_list_spe.pkl')
    words_list_valid = words_list
    # words_list_valid = words_list.loc[words_list.saturation >= 0.9]
    words_list_valid = words_list_valid.loc[words_list_valid.intensity >= 8]

    words_counter = CountVectorizer()
    words_counter = words_counter.fit(words_list_valid.words)
    analyze = words_counter.build_analyzer()
    results = []
    predict_emotion = []
    tweet_id = []
    total_duration = 0

    for index_tweets, tweet in enumerate(train_df.lemmas):
        tweet = ' '.join(unique_list(tweet.split()))
        start = time.time()
        train_sentence = pd.DataFrame()
        for index, word in enumerate(analyze(tweet)):
            result = words_list_valid.loc[words_list_valid.words == word, 'saturation'].to_list()
            if result:
                train_sentence = train_sentence.append(words_list_valid.loc[words_list_valid.words == word,
                                                       'anger':'trust'] * result[0])
            score = train_sentence.sum(axis=0)
        if score.to_list():
            predict_emotion.append(train_sentence.sum(axis=0).idxmax())
            results.append(score.to_list())
        else:
            predict_emotion.append('joy')
            results.append([-1, -1, -1, -1, -1, -1, -1, -1])
        tweet_id.append(train_df.id[index])

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

        print(f'\r{progress:.2f}%, || PT: {td}, ET: {et}, ETA: {eta}', end="")

    output = pd.DataFrame({'id': tweet_id, 'predict_emotion': predict_emotion, 'score': results})
    output['output'] = train_output
    output.to_csv('result.csv')
    output.to_pickle('result.pkl')



    pass


