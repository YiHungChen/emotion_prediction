import datetime as dt
import itertools

import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

from data_preprocessing import load_word_counter, load_word_list
from folder_path import folder_path

from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, ReLU, Softmax
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
import tensorflow as tf
from tensorflow.keras import models,layers

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.layers.recurrent import LSTM
from tensorflow import keras

time_now = dt.datetime.now().strftime("%y-%m-%d_%H%M")

# MetalAM-970
dataFolder = folder_path

from FE_NN import label_encode, label_decode, train_model, Model_score, DataLoader, DLProcess, device, Model_BOW
import os


def plot_confusion_matrix(cm, classes, title='Confusion matrix',
                          cmap=sns.cubehelix_palette(as_cmap=True)):
    classes.sort()
    tick_marks = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes,
           yticklabels=classes,
           title=title,
           xlabel='True label',
           ylabel='Predict label')
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')
        ylim_top = len(classes) - 0.5
        plt.ylim([ylim_top, -.5])
        plt.tight_layout()
        plt.show()


def load_train_data_pure(num_data=None):
    if num_data:
        # DS = pd.read_pickle('train_processed-result/result.pkl')
        DS = pd.read_pickle(f'{dataFolder}Dataset/DS_train.pkl')[:num_data]
    else:
        DS = pd.read_pickle(f'{dataFolder}Dataset/DS_train.pkl')

    msk = np.random.rand(len(DS)) <= 0.8

    train_df = DS[msk].reset_index(drop=True)
    test_df = DS[~msk].reset_index(drop=True)

    # --- training data --- #
    train_input = train_df.lemmas
    train_output = train_df.emotion

    # --- test data --- #
    test_input = test_df.lemmas
    test_output = test_df.emotion

    return train_input, train_output, test_input, test_output


def load_upload_data_pure():
    DS = pd.read_pickle(f'{dataFolder}Dataset/DS.pkl')

    upload_df = DS.loc[DS.identification == 'test']

    # --- training data --- #
    upload_input = upload_df.lemmas

    return upload_df, upload_df


def load_train_data_BOW(num_data=None):
    if num_data:
        # DS = pd.read_pickle('train_processed-result/result.pkl')
        DS = pd.read_pickle(f'{dataFolder}Dataset/DS_train.pkl')[:num_data]
    else:
        DS = pd.read_pickle(f'{dataFolder}Dataset/DS_train.pkl')

    msk = np.random.rand(len(DS)) <= 0.8

    train_df = DS[msk].reset_index(drop=True)
    test_df = DS[~msk].reset_index(drop=True)

    # --- training data --- #
    train_input = train_df.lemmas
    train_output = train_df.emotion

    # --- test data --- #
    test_input = test_df.lemmas
    test_output = test_df.emotion

    word_list_valid = load_word_list(thr_saturation=0.8, thr_intensity=15)
    words_counter, analyze = load_word_counter(word_list_valid)

    train_input = words_counter.transform(train_input)
    test_input = words_counter.transform(test_input)

    print_data_shape(train_input, train_output, test_input, test_output)

    return train_input, train_output, test_input, test_output


def load_train_data(num_data=None):
    if not num_data:
        DS = pd.read_pickle(f'{dataFolder}train_processed-result/result.pkl')
    else:
        DS = pd.read_pickle(f'{dataFolder}train_processed-result/result.pkl')[:num_data]

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


def load_upload_data_BOW():
    DS = pd.read_pickle(f'{dataFolder}Dataset/DS.pkl')
    upload_input = DS.loc[DS.identification == 'test', 'lemmas']
    DS = DS.loc[DS.identification == 'test']

    word_list_valid = load_word_list(thr_saturation=0, thr_intensity=0)
    words_counter, analyze = load_word_counter(word_list_valid)

    upload_input = words_counter.transform(upload_input)

    return DS, upload_input


def load_upload_data():
    DS = pd.read_pickle(f'{dataFolder}test-processed-result/result_test.pkl')
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
    upload_output.to_csv(f'{folder_path}result/DT_result_{time_now}.csv', index=False)


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


def NB_score():
    train_input, train_output, test_input, test_output = load_train_data()

    bnb = BernoulliNB(binarize=0.0)
    bnb.fit(train_input, train_output)

    # --- predict --- #
    train_predict = bnb.predict(train_input)
    test_predict = bnb.predict(test_input)

    print_score(train_output, train_predict, test_output, test_predict)

    upload_df, upload_input = load_upload_data()

    upload_predict = bnb.predict(upload_input)

    output_result(upload_df, upload_predict)

    pass


def NB_BOW():
    train_input, train_output, test_input, test_output = load_train_data_BOW()

    bnb = BernoulliNB(binarize=0.99)
    bnb.fit(train_input, train_output)

    # --- predict --- #
    train_predict = bnb.predict(train_input)
    test_predict = bnb.predict(test_input)

    print_score(train_output, train_predict, test_output, test_predict)

    upload_df, upload_input = load_upload_data_BOW()

    upload_predict = bnb.predict(upload_input)

    output_result(upload_df, upload_predict)

    pass


def GPR_BOW():
    train_input, train_output, test_input, test_output = load_train_data_BOW()

    kernel = DotProduct() + WhiteKernel()

    GPR = GaussianProcessRegressor()
    GPR.fit(train_input, train_output)

    # --- predict --- #
    train_predict = GPR.predict(train_input)
    test_predict = GPR.predict(test_input)

    print_score(train_output, train_predict, test_output, test_predict)

    upload_df, upload_input = load_upload_data_BOW()

    upload_predict = GPR.predict(upload_input)

    output_result(upload_df, upload_predict)

    pass


def NN_score():
    batch_size = 32

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
                model_name=f'{folder_path}model/NN_score_{time_now}.pth')

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

        print(f'\r {i / len(test_dl) * 100:.1f}%', end="")
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
    output.to_csv(f'{folder_path}result/results-Torch-NN-{time_now}.csv', index=False)


def NN_BOW():
    batch_size = 32

    train_input, train_output, test_input, test_output = load_train_data_BOW(5000)

    label_encoder = LabelEncoder()
    label_encoder.fit(train_output)
    numpy.save('classes.npy', label_encoder.classes_)

    train_output = label_encode(label_encoder, train_output)
    test_output = label_encode(label_encoder, test_output)

    TORCH_DS_TRAIN = DLProcess(train_input.toarray(), train_output)
    TORCH_DS_TEST = DLProcess(test_input.toarray(), test_output)

    DL_DS_TRAIN = DataLoader(TORCH_DS_TRAIN, shuffle=True, batch_size=batch_size, drop_last=True)
    DL_DS_TEST = DataLoader(TORCH_DS_TEST, shuffle=True, batch_size=batch_size, drop_last=True)

    num_inputs = train_input.shape[1]
    MD = Model_BOW(num_inputs)

    train_model(x_train=DL_DS_TRAIN,
                x_test=DL_DS_TEST,
                model=MD,
                epoches=30000,
                model_name=f'model/NN_BOW_{time_now}.pth')
    pass


def LG():
    train_input, train_output, test_input, test_output = load_train_data()

    label_encoder = LabelEncoder()
    label_encoder.fit(train_output)
    numpy.save('classes.npy', label_encoder.classes_)

    lg = LogisticRegression(random_state=0).fit(train_input, train_output)
    train_predict = lg.predict(train_input)
    test_predict = lg.predict(test_input)

    print_score(train_output, train_predict, test_output, test_predict)

    upload_df, upload_input = load_upload_data()

    upload_predict = lg.predict(upload_input)

    output_result(upload_df, upload_predict)

    pass


def LG_BOW():
    train_input, train_output, test_input, test_output = load_train_data_BOW()

    label_encoder = LabelEncoder()
    label_encoder.fit(train_output)
    numpy.save('classes.npy', label_encoder.classes_)

    lg = LogisticRegression(C=1, n_jobs=-1, max_iter=100).fit(train_input, train_output)
    train_predict = lg.predict(train_input)
    test_predict = lg.predict(test_input)

    grid = {"C": np.logspace(-3, 3, 7), "penalty": ["l1", "l2"]}  # l1 lasso l2 ridge
    logreg = LogisticRegression()
    logreg_cv = GridSearchCV(logreg, grid, cv=10, scoring="f1_micro")
    logreg_cv.fit(train_input, train_output)

    print("tuned hpyerparameters :(best parameters) ", logreg_cv.best_params_)
    print("accuracy :", logreg_cv.best_score_)
    print('extimator: ', logreg_cv.best_estimator_)

    print_score(train_output, train_predict, test_output, test_predict)

    upload_df, upload_input = load_upload_data_BOW()

    upload_predict = lg.predict(upload_input)

    output_result(upload_df, upload_predict)

    pass

    pass


def NN_BOW_keras():
    batch_size = 16

    train_input, train_output, test_input, test_output = load_train_data_BOW()

    label_encoder = LabelEncoder()
    label_encoder.classes_ = numpy.load('classes.npy', allow_pickle=True)

    train_output = label_encode(label_encoder, train_output)
    test_output = label_encode(label_encoder, test_output)

    input_shape = train_input.shape[1]
    output_shape = train_output.shape[1]

    # --- input layer --- #
    model_input = Input(shape=(input_shape,))
    X = model_input

    # --- 1st hidden layer --- #

    X = Dense(units=64)(X)
    X = ReLU()(X)

    # --- 2nd hidden layer --- #
    X = Dense(units=32)(X)
    X = ReLU()(X)

    # --- output layer --- #
    X = Dense(units=output_shape)(X)
    X = Softmax()(X)

    model_output = X

    # --- create model --- #
    model = Model(inputs=[model_input], outputs=[model_output])

    checkpoint_path = f"{dataFolder}model/model_{time_now}.hdf5"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_best_only=True,
                                                     verbose=1,
                                                     mode='auto',
                                                     period=1,
                                                     monitor='val_loss')

    logdir = os.path.join("logs", time_now)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    # --- loss function & optimizer --- #
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # --- show model construction --- #
    model.summary()

    # --- training setting --- #
    epochs = 1000
    batch_szie = 128

    # --- training process --- #
    history = model.fit(train_input, train_output,
                        epochs=epochs,
                        batch_size=batch_szie,
                        validation_data=[test_input, test_output],
                        callbacks=[cp_callback])


def LSTM_BOW():
    train_input, train_output, test_input, test_output = load_train_data_pure()

    words_lists_valid = load_word_list(thr_saturation=0, thr_intensity=0)
    token = Tokenizer()
    token.fit_on_texts(words_lists_valid.words)

    train_input_seq = token.texts_to_sequences(train_input)
    test_input_seq = token.texts_to_sequences(test_input)

    train_input = sequence.pad_sequences(train_input_seq, maxlen=30)
    test_input = sequence.pad_sequences(test_input_seq, maxlen=30)

    label_encoder = LabelEncoder()
    label_encoder.classes_ = numpy.load('classes.npy', allow_pickle=True)

    train_output = label_encode(label_encoder, train_output)
    test_output = label_encode(label_encoder, test_output)

    model = Sequential()
    model.add(Embedding(output_dim=32,
                        input_dim=len(token.word_index),
                        input_length=30))
    # model.add(Dropout(0.2))
    model.add(LSTM(16))
    # model.add(Dense(units=256, activation='relu'))
    # model.add(Dropout(0.35))
    model.add(Dense(units=8,
                    activation='softmax'))
    model.summary()
    checkpoint_path = f"{dataFolder}model/model_{time_now}.hdf5"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_best_only=True,
                                                     verbose=1,
                                                     mode='auto',
                                                     period=1,
                                                     monitor='val_loss')

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_input, train_output,
                        epochs=30,
                        batch_size=1024,
                        validation_data=[test_input, test_output],
                        callbacks=[cp_callback])

    return model

    pass


def LSTM_BOW_predict(model):
    # model = load_model(f'{folder_path}model/model_21-12-28_2117.hdf5')
    model.summary()
    upload_df, upload_input = load_upload_data_pure()

    train_input, train_output, test_input, test_output = load_train_data_pure()

    words_lists_valid = load_word_list(thr_saturation=0, thr_intensity=0)
    token = Tokenizer()
    token.fit_on_texts(words_lists_valid.words)

    upload_input_seq = token.texts_to_sequences(upload_input.lemmas)

    upload_input = sequence.pad_sequences(upload_input_seq, maxlen=30)

    upload_predict = model.predict(upload_input, batch_size=1024, use_multiprocessing=True)

    label_encoder = LabelEncoder()
    label_encoder.classes_ = numpy.load('classes.npy', allow_pickle=True)

    upload_predict = label_decode(label_encoder, upload_predict)

    output_result(upload_df, upload_predict)

    pass

def CNN():
    # --- load CNN feature --- #
    num_data = 0

    if num_data:
        # DS = pd.read_pickle('train_processed-result/result.pkl')
        DS = pd.read_pickle(f'{dataFolder}CNN_Feature_all.pkl').loc[:num_data]
    else:
        DS = pd.read_pickle(f'{dataFolder}CNN_Feature_all.pkl')

    msk = np.random.rand(len(DS)) <= 0.9

    train_df = DS[msk].reset_index(drop=True)
    test_df = DS[~msk].reset_index(drop=True)

    train_input = train_df.CNN_Feature
    train_output = train_df.predict_emotion

    test_input = test_df.CNN_Feature
    test_output = test_df.predict_emotion

    label_encoder = LabelEncoder()
    label_encoder.classes_ = numpy.load('classes.npy', allow_pickle=True)

    train_output = label_encode(label_encoder, train_output)
    test_output = label_encode(label_encoder, test_output)

    model = models.Sequential()
    model.add(layers.Conv2D(64, (2, 2), padding="same",activation='selu', input_shape=(30, 8, 1)))
    model.add(Dropout(0.4))
    # model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (2, 2), padding="same", activation='selu'))
    model.add(layers.Conv2D(256, (2, 2), padding="same", activation='selu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))
    model.add(layers.Flatten())
    model.add(Dropout(0.4))
    model.add(layers.Dense(128, activation='selu'))
    model.add(Dropout(0.4))
    model.add(layers.Dense(8, activation='softmax'))

    model.summary()

    opt = keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    train_input = np.array(train_input.tolist())
    train_output =  np.array(train_output.tolist())

    test_input = np.array(test_input.tolist())
    test_output = np.array(test_output.tolist())

    checkpoint_path = f"{dataFolder}model/model_CNN_{time_now}.hdf5"

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_best_only=True,
                                                     verbose=1,
                                                     mode='auto',
                                                     period=1,
                                                     monitor='val_loss')

    history = model.fit(train_input,
                        train_output,
                        epochs=1000,
                        batch_size=256,
                        validation_data=[test_input, test_output],
                        callbacks=[cp_callback])







    pass



if __name__ == '__main__':
    # NN_score()

    # NN_score_predict()

    # Decision_tree()

    # NN_BOW()

    # LG()

    # LG_BOW()

    # NB_score()

    # NB_BOW()

    # NN_BOW_keras()

    # model = LSTM_BOW()

    # LSTM_BOW_predict(model)

    CNN()

    pass
