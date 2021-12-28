"""
This program is for the separation of the training and testing data. The data with different emotion will also be
separated in this program
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from folder_path import folder_path


def myRange(start,end,step):
    i = start
    while i < end:
        yield i
        i += step
    yield end


def calculate_frequency(train, emotion, min_emotion_value, total_words_df):
    # --- joy intensity --- #
    num_emotion = len(train.loc[train.emotion == emotion])
    emotion_frequencies = pd.DataFrame({'words': BOW_vectorizer.get_feature_names_out()})
    last_batch = 0
    for i in myRange(min_emotion_value, num_emotion, min_emotion_value):
        rest = min_emotion_value - (i - last_batch)
        emotion_df = train.loc[train.emotion == emotion].sample(n=min_emotion_value)
        if rest:
            emotion_df = emotion_df.append(train.loc[train.emotion == emotion][0:last_batch].sample(n=rest))
            pass

        emotion_counts = BOW_vectorizer.transform(emotion_df.lemmas)
        # print(train_joy_counts.shape)
        emotion_frequencies[i] = np.asarray(emotion_counts.sum(axis=0))[0]
        last_batch = i
        pass
    total_words_df[emotion] = emotion_frequencies.max(axis=1, numeric_only=True)
    print(f'most {emotion} word: [{BOW_vectorizer.get_feature_names_out()[np.argmax(total_words_df[emotion])]}] '
          f'with frequency of {max(total_words_df[emotion])}')
    return total_words_df


# --- main --- #
if __name__ == '__main__':
    # --- load dataset --- #
    train = pd.read_pickle(f'{folder_path}Dataset/DS_train.pkl')

    # --- calculate the histogram --- #
    post_total = len(train)
    emotion_value = train.groupby(['emotion']).count()['text']
    min_emotion_value = min(emotion_value)
    emotion_value = emotion_value.apply(lambda x: round(x * 100 / post_total, 3))

    # --- create total words list --- #
    BOW_vectorizer = CountVectorizer(stop_words='english')
    BOW_vectorizer.fit(train.lemmas)

    total_words_list = BOW_vectorizer.get_feature_names_out()
    # total_words_list_lemmas = lemmas(total_words_list)
    total_words_df = pd.DataFrame()
    total_words_df['words'] = total_words_list
    # total_words_df['lemmas'] = total_words_list_lemmas

    total_words_df = calculate_frequency(train, 'anger', min_emotion_value, total_words_df)
    total_words_df = calculate_frequency(train, 'joy', min_emotion_value, total_words_df)
    total_words_df = calculate_frequency(train, 'anticipation', min_emotion_value, total_words_df)
    total_words_df = calculate_frequency(train, 'disgust', min_emotion_value, total_words_df)
    total_words_df = calculate_frequency(train, 'fear', min_emotion_value, total_words_df)
    total_words_df = calculate_frequency(train, 'sadness', min_emotion_value, total_words_df)
    total_words_df = calculate_frequency(train, 'surprise', min_emotion_value, total_words_df)
    total_words_df = calculate_frequency(train, 'trust', min_emotion_value, total_words_df)

    total_words_df.to_csv(f'{folder_path}words_list/words_list.csv')
    total_words_df.to_pickle(f'{folder_path}words_list/words_list.pkl')

    pass
