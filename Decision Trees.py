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

from data_preprocessing import load_word_counter, load_word_list
from folder_path import folder_path

time_now = dt.datetime.now().strftime("%y-%m-%d_%H%M")

# MetalAM-970
dataFolder = folder_path

from FE_NN import label_encode, label_decode, train_model, Model_score, DataLoader, DLProcess, device


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


def load_train_data_BOW(num_data=None):
    if not num_data:
        # DS = pd.read_pickle('train_processed-result/result.pkl')
        DS = pd.read_pickle(f'{dataFolder}Dataset/DS_train.pkl')[:num_data]
    else:
        DS = pd.read_pickle(f'{dataFolder}Dataset/DS_train.pkl')

    msk = np.random.rand(len(DS)) <= 0.8

    train_df = DS[msk].reset_index(drop=True)
    test_df = DS[msk].reset_index(drop=True)

    # --- training data --- #
    train_input = train_df.lemmas
    train_output = train_df.emotion

    # --- test data --- #
    test_input = test_df.lemmas
    test_output = test_df.emotion

    word_list_valid = load_word_list(thr_saturation=0, thr_intensity=0)
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
    output.to_csv(f'result/results-Torch-NN-{time_now}.csv', index=False)


def NN_BOW():
    batch_size = 128

    train_input, train_output, test_input, test_output = load_train_data_BOW()

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
    MD = Model_score(num_inputs)

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

    lg = LogisticRegression(C=1000, n_jobs=-1, max_iter=1000).fit(train_input, train_output)
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


if __name__ == '__main__':
    # NN_score()

    # NN_score_predict()

    # Decision_tree()

    # NN_BOW()

    # LG()

    LG_BOW()

    pass
