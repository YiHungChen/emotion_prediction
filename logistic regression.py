from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

import keras
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

def plot_confusion_matrix(cm, classes, title='Confusion matrix',
                          cmap= sns.cubehelix_palette(as_cmap=True)):
    classes.sort()
    tick_marks = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes,
           yticklabels=classes,
           title=title,
           xlabel='True label',
           ylabel='Predict label')
    fmt='d'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')
        ylim_top = len(classes) -0.5
        plt.ylim([ylim_top, -.5])
        plt.tight_layout()
        plt.show()

# HSV feature with Decision Tree
"""
DS = pd.read_pickle('result_all.pkl')
msk = np.random.rand(len(DS)) <= 0.8

train_df = DS[msk].reset_index(drop=True)
test_df = DS[~msk].reset_index(drop=True)


# --- training data --- #
train_input = pd.DataFrame(train_df.score.to_list())
train_output = train_df.output

# --- test data --- #
test_input = pd.DataFrame(test_df.score.to_list())
test_output = test_df.output


# --- print shape of the data --- #
print(f'Training input shape: {train_input.shape}')
print(f'Training output shape: {train_output.shape}')
print(f'Testing input shape: {valid_input.shape}')
print(f'Testing output shape: {valid_output.shape}')

# --- build decision tree model --- #
DT_model = DecisionTreeClassifier(random_state = 0)

# --- model training --- #
DT_model = DT_model.fit(train_input, train_output)

# --- predict --- #
train_predict= DT_model.predict(train_input)
test_predict = DT_model.predict(valid_input)

# --- show the prediction results --- #
test_predict[:10]

train_acc = accuracy_score(y_true = train_output, y_pred = train_predict)
test_acc = accuracy_score(y_true = valid_output, y_pred = test_predict)

print(f'training accuracy: {train_acc:.2f}')
print(f'testing accuracy: {test_acc:.2f}')
print(classification_report(y_true= valid_output, y_pred=test_predict))
"""

# BOW with pre-process feature goes to 0.29

DS = pd.read_pickle('Dataset/DS_train.pkl')[:50000]

msk = np.random.rand(len(DS)) <= 0.8

train_df = DS[msk].reset_index(drop=True)
test_df = DS[~msk].reset_index(drop=True)

words_list = pd.read_pickle('hsv/hsv_list_spe.pkl')

words_list_valid = words_list
words_list_valid = words_list.loc[words_list.saturation >= 0.4]
words_list_valid = words_list_valid.loc[words_list_valid.intensity >= 8]
print(f'valid words of {len(words_list_valid)}')


# --- build analyzers bag of words --- #
BOW_500 = CountVectorizer(max_features=1000, tokenizer=nltk.word_tokenize)

# --- apply analyzer to training data --- #
BOW_500.fit(train_df['lemmas'])

train_BOW_features_500 = BOW_500.transform(train_df['lemmas'])

# --- check dimension --- #
print(train_BOW_features_500.shape)

feature_names_500 = BOW_500.get_feature_names()
print(feature_names_500[100:110])

# --- training Data ---#
train_input = BOW_500.transform(train_df.lemmas)
train_output = train_df.emotion

# --- test Data --- #
test_input = BOW_500.transform(test_df.lemmas)
test_output = test_df.emotion
"""
# --- print shape of the data --- #
print(f'Training input shape: {train_input.shape}')
print(f'Training output shape: {train_output.shape}')
print(f'Testing input shape: {test_input.shape}')
print(f'Testing output shape: {test_output.shape}')
"""

"""
# --- build decision tree model --- #
DT_model = DecisionTreeClassifier(random_state = 0)

# --- model training --- #
DT_model = DT_model.fit(train_input, train_output)

# --- predict --- #
train_predict= DT_model.predict(train_input)
test_predict = DT_model.predict(test_input)


# --- show the prediction results --- #
test_predict[:10]

train_acc = accuracy_score(y_true = train_output, y_pred = train_predict)
test_acc = accuracy_score(y_true = test_output, y_pred = test_predict)

print(f'training accuracy: {train_acc:.2f}')
print(f'testing accuracy: {test_acc:.2f}')
print(classification_report(y_true= test_output, y_pred=test_predict))
"""

label_encoder = LabelEncoder()
label_encoder.fit(train_output)


print(f'check label: {label_encoder.classes_}')
print(f'--- Before convert ---')
print(f'first 4 output label: \n{train_output[:4]}\n'
      f'shape of the training output: {train_output.shape}\n'
      f'shape of the testing output : {test_output.shape}')

# --- function for label encoding --- #
def label_encode(le, labels):
    enc = le.transform(labels)
    return keras.utils.np_utils.to_categorical(enc)


def label_decode(le, one_hot_label):
    dec = np.argmax(one_hot_label, axis=1)
    return le.inverse_transform(dec)


train_output = label_encode(label_encoder, train_output)
test_output = label_encode(label_encoder, test_output)

print(f'--- After convert ---')
print(f'first 4 output label: \n{train_output[:4]}\n'
      f'shape of the training output: {train_output.shape}\n'
      f'shape of the testing output : {test_output.shape}')

# --- check dimesion of the input and output --- #
input_shape = train_input.shape[1]
output_shape = train_output.shape[1]

print(f'input shape of : {input_shape}\n'
      f'output shape of: {output_shape}')


lm = LinearRegression()
lm.fit(train_input, train_output)

pred_result = lm.predict(test_input)

print(f'test accuracy: {accuracy_score(label_decode(label_encoder, test_output), label_decode(label_encoder,pred_result)):.2f}')


# --- make prediction --- #

DS_test = pd.read_pickle('Dataset/DS.pkl')
upload_df = DS_test.loc[DS_test.identification == 'test']
upload_input = BOW_500.transform(upload_df.lemmas)
# upload_predict = DT_model.predict(upload_input)
# --- Using Score as feature --- #
# DS_score = pd.read_pickle('result_test.pkl')
# upload_input = pd.DataFrame(DS_score.score.to_list())
upload_predict = lm.predict(upload_input)
upload_predict = label_decode(label_encoder, upload_predict)
upload_output = pd.DataFrame({'id':upload_df.id, 'emotion': upload_predict})
upload_output.to_csv('results-DT-1223.csv', index=False)