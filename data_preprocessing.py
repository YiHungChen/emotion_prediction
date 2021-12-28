import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import time
from folder_path import folder_path


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
    words_counter = CountVectorizer(stop_words='english')
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
        tweet = ' '.join(unique_list(tweet.split()))

        # --- start time counting --- #
        start = time.time()

        train_sentence = words_list_valid.loc[words_list_valid.words.isin(analyze(tweet)), 'anger':'trust']
        train_saturation = words_list_valid.loc[words_list_valid.words.isin(analyze(tweet)), 'saturation']
        train_sentence = train_sentence.mul(train_saturation, axis=0)

        score = train_sentence.sum(axis=0)
        score = normalization(score)

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
    words_list_valid = load_word_list(thr_saturation=0.5, thr_intensity=0)



    # --- score calculation --- #
    output = score_calculation(train_df, words_list_valid)

    # --- output file --- #
    train_output = train_df.emotion
    output['output'] = train_output
    output.to_csv('result.csv')
    output.to_pickle('result.pkl')

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


if __name__ == '__main__':

    training_data()

    # test_data()

    pass
