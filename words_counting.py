"""
This program is for the separation of the training and testing data. The data with different emotion will also be
separated in this program
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from lemmatization import lemmas_sentence
from nltk.tokenize.treebank import TreebankWordDetokenizer

# --- main --- #
if __name__ == '__main__':
    # --- load dataset --- #
    train = pd.read_pickle('Dataset/DS_train.pkl')

    # --- calculate the histogram --- #
    post_total = len(train)
    emotion_value = train.groupby(['emotion']).count()['text']
    min_emotion_value = min(emotion_value)
    emotion_value = emotion_value.apply(lambda x: round(x * 100 / post_total, 3))

    # --- plot --- #
    figure, axis = plt.subplots()
    plt.bar(emotion_value.index, emotion_value.values)

    # --- rearrange labels --- #
    plt.ylabel('% of instances')
    plt.xlabel('Emotion')
    plt.title('Emotion distribution')
    plt.grid(True)
    plt.show()

    # --- create total words list --- #
    BOW_vectorizer = CountVectorizer()
    BOW_vectorizer.fit(train.lemmas)

    total_words_list = BOW_vectorizer.get_feature_names_out()
    # total_words_list_lemmas = lemmas(total_words_list)
    total_words_df = pd.DataFrame()
    total_words_df['words'] = total_words_list
    # total_words_df['lemmas'] = total_words_list_lemmas

    train_anger_df = train.loc[train.emotion == 'anger'].sample(n=min_emotion_value)
    train_joy_df = train.loc[train.emotion == 'joy'].sample(n=min_emotion_value)
    train_anticipation_df = train.loc[train.emotion == 'anticipation'].sample(n=min_emotion_value)
    train_disgust_df = train.loc[train.emotion == 'disgust'].sample(n=min_emotion_value)
    train_fear_df = train.loc[train.emotion == 'fear'].sample(n=min_emotion_value)
    train_sadness_df = train.loc[train.emotion == 'sadness'].sample(n=min_emotion_value)
    train_surprise_df = train.loc[train.emotion == 'surprise'].sample(n=min_emotion_value)
    train_trust_df = train.loc[train.emotion == 'trust'].sample(n=min_emotion_value)

    # --- anger intensity --- #
    train_anger_counts = BOW_vectorizer.transform(train_anger_df.lemmas)
    print(train_anger_counts.shape)
    train_anger_frequencies = []
    train_anger_frequencies = np.asarray(train_anger_counts.sum(axis=0))[0] / len(train_anger_df)
    total_words_df['anger'] = train_anger_frequencies
    print(f'most anger word: [{BOW_vectorizer.get_feature_names_out()[np.argmax(train_anger_frequencies)]}] '
          f'with frequency of {max(train_anger_frequencies)}')

    # --- joy intensity --- #
    train_joy_counts = BOW_vectorizer.transform(train_joy_df.lemmas)
    print(train_joy_counts.shape)
    train_joy_frequencies = []
    train_joy_frequencies = np.asarray(train_joy_counts.sum(axis=0))[0] / len(train_joy_df)
    total_words_df['joy'] = train_joy_frequencies
    print(f'most joy word: [{BOW_vectorizer.get_feature_names_out()[np.argmax(train_joy_frequencies)]}] '
          f'with frequency of {max(train_joy_frequencies)}')

    # --- anticipation intensity --- #
    train_anticipation_counts = BOW_vectorizer.transform(train_anticipation_df.lemmas)
    print(train_anticipation_counts.shape)
    train_anticipation_frequencies = []
    train_anticipation_frequencies = np.asarray(train_anticipation_counts.sum(axis=0))[0] / len(train_anticipation_df)
    total_words_df['anticipation'] = train_anticipation_frequencies
    print(
        f'most anticipation word: [{BOW_vectorizer.get_feature_names_out()[np.argmax(train_anticipation_frequencies)]}] '
        f'with frequency of {max(train_anticipation_frequencies)}')

    # --- disgust intensity --- # 
    train_disgust_counts = BOW_vectorizer.transform(train_disgust_df.lemmas)
    print(train_disgust_counts.shape)
    train_disgust_frequencies = []
    train_disgust_frequencies = np.asarray(train_disgust_counts.sum(axis=0))[0] / len(train_disgust_df)
    total_words_df['disgust'] = train_disgust_frequencies
    print(f'most disgust word: [{BOW_vectorizer.get_feature_names_out()[np.argmax(train_disgust_frequencies)]}] '
          f'with frequency of {max(train_disgust_frequencies)}')

    # --- fear intensity --- # 
    train_fear_counts = BOW_vectorizer.transform(train_fear_df.lemmas)
    print(train_fear_counts.shape)
    train_fear_frequencies = []
    train_fear_frequencies = np.asarray(train_fear_counts.sum(axis=0))[0] / len(train_fear_df)
    total_words_df['fear'] = train_fear_frequencies
    print(f'most fear word: [{BOW_vectorizer.get_feature_names_out()[np.argmax(train_fear_frequencies)]}] '
          f'with frequency of {max(train_fear_frequencies)}')

    # --- sadness intensity --- # 
    train_sadness_counts = BOW_vectorizer.transform(train_sadness_df.lemmas)
    print(train_sadness_counts.shape)
    train_sadness_frequencies = []
    train_sadness_frequencies = np.asarray(train_sadness_counts.sum(axis=0))[0] / len(train_sadness_df)
    total_words_df['sadness'] = train_sadness_frequencies
    print(f'most sadness word: [{BOW_vectorizer.get_feature_names_out()[np.argmax(train_sadness_frequencies)]}] '
          f'with frequency of {max(train_sadness_frequencies)}')

    # --- surprise intensity --- # 
    train_surprise_counts = BOW_vectorizer.transform(train_surprise_df.lemmas)
    print(train_surprise_counts.shape)
    train_surprise_frequencies = []
    train_surprise_frequencies = np.asarray(train_surprise_counts.sum(axis=0))[0] / len(train_surprise_df)
    total_words_df['surprise'] = train_surprise_frequencies
    print(f'most surprise word: [{BOW_vectorizer.get_feature_names_out()[np.argmax(train_surprise_frequencies)]}] '
          f'with frequency of {max(train_surprise_frequencies)}')

    # --- trust intensity --- # 
    train_trust_counts = BOW_vectorizer.transform(train_trust_df.lemmas)
    print(train_trust_counts.shape)
    train_trust_frequencies = []
    train_trust_frequencies = np.asarray(train_trust_counts.sum(axis=0))[0] / len(train_trust_df)
    total_words_df['trust'] = train_trust_frequencies
    print(f'most trust word: [{BOW_vectorizer.get_feature_names_out()[np.argmax(train_trust_frequencies)]}] '
          f'with frequency of {max(train_trust_frequencies)}')

    total_words_df.to_csv('words_list.csv')
    total_words_df.to_pickle('Dataset/words_list.pkl')

    pass
