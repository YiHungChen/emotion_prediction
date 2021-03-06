import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import time
from folder_path import folder_path
import numpy as np

def unique_list(l):
    ulist = []
    [ulist.append(x) for x in l if x not in ulist]
    return ulist


def load_data(num_data=0):
    # dataset_df = pd.read_pickle('Dataset/DS_train.pkl').sample(n=100000).reset_index(drop=True)
    if not num_data:
        dataset_df = pd.read_pickle(f'{folder_path}Dataset/DS_train.pkl')
    else:
        dataset_df = pd.read_pickle(f'{folder_path}Dataset/DS_train.pkl')[:num_data]

    train_df = dataset_df

    return train_df


def load_data_test():
    # dataset_df = pd.read_pickle('Dataset/DS_train.pkl').sample(n=100000).reset_index(drop=True)

    dataset_df = pd.read_pickle(f'{folder_path}Dataset/DS.pkl')
    dataset_df = dataset_df.loc[dataset_df.identification == 'test'].reset_index(drop=True)

    test_df = dataset_df
    return test_df


def load_word_list(thr_saturation=0, thr_intensity=0):
    words_list = pd.read_pickle(f'{folder_path}hsv/hsv_list_spe.pkl')
    words_list_valid = words_list
    words_list_valid = words_list.loc[words_list.saturation >= thr_saturation]
    words_list_valid = words_list_valid.loc[words_list_valid.intensity >= thr_intensity]

    return words_list_valid

def load_word_counter(words_list_valid: pd.DataFrame()):
    words_counter = CountVectorizer()
    words_counter = words_counter.fit(words_list_valid.words)
    analyze = words_counter.build_analyzer()

    return words_counter, analyze


def score_calculation(train_df, words_list_valid):

    # --- initiate the parameters --- #
    results = []
    predict_emotion = []
    tweet_id = []
    total_duration = 0

    # --- initiate the word counter --- #
    words_counter, analyze = load_word_counter(words_list_valid)

    for index_tweets, tweet in enumerate(train_df.lemmas):

        # --- clean the repeated words --- #
        # tweet = ' '.join(unique_list(tweet.split()))

        # --- start time counting --- #
        start = time.time()

        train_sentence = words_list_valid.loc[words_list_valid.words.isin(analyze(tweet)), 'anger':'trust']
        train_saturation = words_list_valid.loc[words_list_valid.words.isin(analyze(tweet)), 'saturation']
        train_sentence = train_sentence.mul(train_saturation, axis=0)

        score = train_sentence.sum(axis=0)
        # score = normalization(score)

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

    return output

    pass


def CNN_data_preparation(train_df, words_list_valid):

    results = []
    predict_emotion = []
    total_duration = 0
    words_counter, analyze = load_word_counter(words_list_valid)
    columns = ['anger', 'joy', 'anticipation', 'disgust', 'fear', 'sadness', 'surprise', 'trust', 'saturation']

    for index_tweet, tweet in enumerate(train_df.lemmas):
        start = time.time()
        word_list = words_list_valid.words.isin(analyze(tweet))
        train_sentence = words_list_valid.loc[word_list, columns]
        train_output = train_sentence.mul(train_sentence['saturation'], axis=0).reset_index(drop=True).loc[:, 'anger':'trust']
        zero_df = pd.DataFrame(np.zeros([30-len(train_output), 8]), columns=train_output.columns)
        train_output = train_output.append(zero_df)
        results.append(train_output.values)
        predict_emotion.append(train_df.emotion.loc[index_tweet])

        end = time.time()
        duration = end - start
        total_duration += duration
        progress = ((index_tweet + 1) / len(train_df)) * 100

        et = total_duration * 100 / progress
        td = total_duration
        eta = et - td

        et = time.strftime('%H:%M:%S', time.gmtime(total_duration * 100 / progress))
        td = time.strftime('%H:%M:%S', time.gmtime(total_duration))
        eta = time.strftime('%H:%M:%S', time.gmtime(eta))

        print(f'\r{progress:.2f}%, || PT: {td}, ET: {et}, ETA: {eta},'
              f' {(total_duration/(index_tweet+1)):.4f} s/tweet', end="")

        pass
    output = pd.DataFrame({'predict_emotion': predict_emotion, 'CNN_Feature': results})
    return output

    pass

def CNN_data_preparation_test(test_df, words_list_valid):

    results = []
    predict_id = []
    total_duration = 0
    words_counter, analyze = load_word_counter(words_list_valid)
    columns = ['anger', 'joy', 'anticipation', 'disgust', 'fear', 'sadness', 'surprise', 'trust', 'saturation']

    for index_tweet, tweet in enumerate(test_df.lemmas):
        start = time.time()
        word_list = words_list_valid.words.isin(analyze(tweet))
        train_sentence = words_list_valid.loc[word_list, columns]
        train_output = train_sentence.mul(train_sentence['saturation'], axis=0).reset_index(drop=True).loc[:, 'anger':'trust']
        zero_df = pd.DataFrame(np.zeros([30-len(train_output), 8]), columns=train_output.columns)
        train_output = train_output.append(zero_df)
        results.append(train_output.values)
        predict_id.append(test_df.id.loc[index_tweet])

        end = time.time()
        duration = end - start
        total_duration += duration
        progress = ((index_tweet + 1) / len(test_df)) * 100

        et = total_duration * 100 / progress
        td = total_duration
        eta = et - td

        et = time.strftime('%H:%M:%S', time.gmtime(total_duration * 100 / progress))
        td = time.strftime('%H:%M:%S', time.gmtime(total_duration))
        eta = time.strftime('%H:%M:%S', time.gmtime(eta))

        print(f'\r{progress:.2f}%, || PT: {td}, ET: {et}, ETA: {eta},'
              f' {(total_duration/(index_tweet+1)):.4f} s/tweet', end="")

        pass
    output = pd.DataFrame({'predict_emotion': predict_id, 'CNN_Feature': results})
    return output

    pass


def normalization(df):

    a = df.loc['anger':'trust']
    b = a.max()
    c = a.min()
    d = (a-c)/(b-c)
    df.loc['anger':'trust'] = d
    return df


def training_data():
    # --- load data --- #
    train_df = load_data()

    # --- load word list --- #
    words_list_valid = load_word_list(thr_saturation=0, thr_intensity=0)


    # --- score calculation --- #
    output = score_calculation(train_df, words_list_valid)

    # --- output file --- #
    train_output = train_df.emotion
    output['output'] = train_output
    output.to_csv(f'{folder_path}result\result.csv')
    output.to_pickle(f'{folder_path}result\result.pkl')

    pass


def test_data():
    # --- load data --- #
    test_df = load_data_test()

    # --- load word list --- #
    words_list_valid = load_word_list(thr_saturation=0, thr_intensity=8)

    # --- score calculation --- #
    output = score_calculation(test_df, words_list_valid)

    # --- output file --- #
    output.to_csv(f'{folder_path}result_test.csv')
    output.to_pickle(f'{folder_path}result_test.pkl')


def training_data_CNN():
    # --- load data --- #
    train_df = load_data()

    # --- load word list --- #
    words_list_valid = load_word_list(thr_saturation=0, thr_intensity=0)

    # --- CNN dataset preparation --- #
    output = CNN_data_preparation(train_df, words_list_valid)
    output.to_pickle(f'{folder_path}CNN_Feature-1.pkl')
    output.to_csv(f'{folder_path}CNN_Feature-1.csv', index=False)



    pass


def test_data_CNN():
    # --- load data --- #
    test_df = load_data_test()

    # --- load word list --- #
    words_list_valid = load_word_list(thr_saturation=0, thr_intensity=0)

    # --- CNN dataset preparation --- #
    output = CNN_data_preparation_test(test_df, words_list_valid)
    output.to_pickle(f'{folder_path}CNN_Feature_test.pkl')
    output.to_csv(f'{folder_path}CNN_Feature_test.csv', index=False)



    pass


if __name__ == '__main__':

    # training_data()

    # test_data()

    training_data_CNN()

    # test_data_CNN()

    pass
