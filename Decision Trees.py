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
import datetime as dt
import numpy
import torch

time_now = dt.datetime.now().strftime("%y-%m-%d_%H%M")

from FE_NN import label_encode, label_decode, train_model, Model_score, DataLoader, DLProcess, device

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


def load_train_data(num_data=None):
    if not num_data:
        DS = pd.read_pickle('train_processed-result/result.pkl')
    else:
        DS = pd.read_pickle('train_processed-result/result.pkl')[:num_data]

    msk = np.random.rand(len(DS)) <= 0.8

    train_df = DS[msk].reset_index(drop=True)
    test_df = DS[~msk].reset_index(drop=True)

    # --- training data --- #
    train_input = pd.DataFrame(train_df.score.to_list())
    train_output = train_df.output

    # --- test data --- #
    test_input = pd.DataFrame(test_df.score.to_list())
    test_output = test_df.output

    print_data_shape(train_input, train_output, test_input, test_output)

    return train_input, train_output, test_input, test_output


def load_upload_data():
    DS = pd.read_pickle('test_processed-result/result_test.pkl')
    upload_input = pd.DataFrame(DS.score.to_list())

    return DS, upload_input


def print_data_shape(train_input, train_output, test_input, test_output):
    print(f'Training input shape: {train_input.shape}')
    print(f'Training output shape: {train_output.shape}')
    print(f'Testing input shape: {test_input.shape}')
    print(f'Testing output shape: {test_output.shape}')
    pass


def DT_model_func(train_input, train_output):
    # --- build decision tree model --- #
    DT_model = DecisionTreeClassifier(random_state=0)

    # --- model training --- #
    DT_model = DT_model.fit(train_input, train_output)

    return DT_model


def print_score(train_output, train_predict, test_output, test_predict):
    train_acc = accuracy_score(y_true=train_output, y_pred=train_predict)
    test_acc = accuracy_score(y_true=test_output, y_pred=test_predict)

    print(f'training accuracy: {train_acc:.2f}')
    print(f'testing accuracy: {test_acc:.2f}')
    print(classification_report(y_true=test_output, y_pred=test_predict))
    pass


def output_result(upload_df, upload_predict):
    upload_output = pd.DataFrame({'id': upload_df.id, 'emotion': upload_predict})
    upload_output.to_csv(f'result/DT_result_{time_now}.csv', index=False)


def Decision_tree():

    # HSV feature with Decision Tree
    train_input, train_output, test_input, test_output = load_train_data()

    DT_model = DT_model_func(train_input, train_output)

    # --- predict --- #
    train_predict = DT_model.predict(train_input)
    test_predict = DT_model.predict(test_input)

    print_score(train_output, train_predict, test_output, test_predict)

    upload_df, upload_input = load_upload_data()

    upload_predict = DT_model.predict(upload_input)

    output_result(upload_df, upload_predict)

    pass


def NN_score():
    batch_size = 128

    train_input, train_output, test_input, test_output = load_train_data()

    label_encoder = LabelEncoder()
    label_encoder.fit(train_output)
    numpy.save('classes.npy', label_encoder.classes_)

    train_output = label_encode(label_encoder, train_output)
    test_output = label_encode(label_encoder, test_output)

    TORCH_DS_TRAIN = DLProcess(train_input.values, train_output)
    TORCH_DS_TEST = DLProcess(test_input.values, test_output)

    DL_DS_TRAIN = DataLoader(TORCH_DS_TRAIN, shuffle=True, batch_size=batch_size, drop_last=True)
    DL_DS_TEST = DataLoader(TORCH_DS_TEST, shuffle=True, batch_size=batch_size, drop_last=True)

    num_inputs = train_input.shape[1]
    MD = Model_score(num_inputs)

    train_model(x_train=DL_DS_TRAIN,
                x_test=DL_DS_TEST,
                model=MD,
                epoches=30000,
                model_name=f'model/NN_score_{time_now}.pth')

    pass


# --- target the model --- #
def target_model(test_dl, model):
    predictions = list()
    for i, (inputs, targets) in enumerate(test_dl):
        # --- evaluate the model on the test set --- #
        yhat = model(inputs.to(device))
        # --- retrieve numpy array --- #
        yhat_cpu = yhat.cpu()

        yhat = yhat_cpu.detach().numpy()

        yhat = yhat.reshape((len(yhat), 8))
        # --- store --- #
        predictions.append(yhat)
        # id.append(targets)

        print(f'\r {i/len(test_dl)*100:.1f}%', end="")
    print()

    predictions = np.vstack(predictions)
    return predictions


def NN_score_predict():
    batch_size = 1

    DS, upload_input = load_upload_data()
    upload_output = [0] * len(upload_input)

    label_encoder = LabelEncoder()
    label_encoder.classes_ = numpy.load('classes.npy', allow_pickle=True)

    TORCH_DS_UPLOAD = DLProcess(upload_input.values, upload_output)
    DL_DS_UPLOAD = DataLoader(TORCH_DS_UPLOAD, shuffle=False, batch_size=1, drop_last=True)

    MD = torch.load('model/NN_score_21-12-26_1619.pth')

    predictions = target_model(DL_DS_UPLOAD, MD)

    pred = label_decode(label_encoder, predictions)

    output = pd.DataFrame({'id': DS.id, 'emotion': pred})
    output.to_csv(f'result/results-Torch-NN-{time_now}.csv', index=False)


if __name__ == '__main__':
    NN_score()

    # NN_score_predict()

    # Decision_tree()


    pass





# BOW with pre-process feature goes to 0.29
"""
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

# --- build models --- #
from keras.models import Model
from keras.layers import Input, Dense, ReLU, Softmax, Dropout

# --- input layer --- #
model_input = Input(shape=(input_shape, ))
X = model_input
# --- 2nd hidden layer --- #
X = Dense(units=16)(X)
X = ReLU()(X)
X = Dropout(0.2)(X)

X = Dense(units=16)(X)
X = ReLU()(X)
X = Dropout(0.2)(X)

# --- 2nd hidden layer --- #
X = Dense(units=8)(X)
X = ReLU()(X)
X = Dropout(0.2)(X)

# --- output layer --- #
X = Dense(units=output_shape)(X)
X = Softmax()(X)

model_output = X

# --- create model --- #
model = Model(inputs=[model_input], outputs=[model_output])

# --- loss function & optimizer --- #
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --- show model construction --- #
model.summary()

from keras.callbacks import CSVLogger, ModelCheckpoint
csv_logger = CSVLogger('logs/training_log.csv')

# --- training setting --- #
epochs = 1000
batch_szie = 128
tensorboard_callback = tf.keras.callbacks.TensorBoard("logs", histogram_freq=1)
filepath = "model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# --- training process --- #
history = model.fit(train_input, train_output,
                    epochs=epochs,
                    batch_size=batch_szie,
                    validation_data=[test_input, test_output],
                    callbacks=[tensorboard_callback])
print(f'training finished')

# --- prediction --- #
pred_result = model.predict(test_input, batch_size=128)
pred_result[:5]

pred_result = label_decode(label_encoder, pred_result)
pred_result[:5]

print(f'test accuracy: {accuracy_score(label_decode(label_encoder, test_output), pred_result):.2f}')


# --- make prediction --- #

DS_test = pd.read_pickle('Dataset/DS.pkl')
upload_df = DS_test.loc[DS_test.identification == 'test']
upload_input = BOW_500.transform(upload_df.lemmas)
# upload_predict = DT_model.predict(upload_input)
# --- Using Score as feature --- #
# DS_score = pd.read_pickle('result_test.pkl')
# upload_input = pd.DataFrame(DS_score.score.to_list())
upload_predict = model.predict(upload_input, batch_size=128)
upload_predict = label_decode(label_encoder, upload_predict)
upload_output = pd.DataFrame({'id':upload_df.id, 'emotion': upload_predict})
upload_output.to_csv('results-DT-1223.csv', index=False)
"""
