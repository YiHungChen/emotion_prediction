""" This python file is for using NN """

# --- import libraries --- #
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.nn.init import kaiming_uniform_
from torch.utils.data import DataLoader, Dataset


from torch.utils.tensorboard import SummaryWriter
import datetime as dt

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import keras

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
time_now = dt.datetime.now().strftime("%Y%m%d")

# --- end of libraries --- #

str_load_process = ['[>-------------------]', '[=>------------------]', '[==>-----------------]',
                    '[===>----------------]', '[====>---------------]', '[=====>--------------]',
                    '[======>-------------]', '[=======>------------]', '[========>-----------]',
                    '[=========>----------]', '[==========>---------]', '[===========>--------]',
                    '[============>-------]', '[=============>------]', '[==============>-----]',
                    '[===============>----]', '[================>---]', '[=================>--]',
                    '[==================>-]', '[===================>]', '[====================]']

tensorboard_model_name = dt.datetime.now().strftime("%Y%m%d")

# --- function for label encoding --- #
def label_encode(le, labels):
    enc = le.transform(labels)
    return keras.utils.np_utils.to_categorical(enc)


def label_decode(le, one_hot_label):
    dec = np.argmax(one_hot_label, axis=1)
    return le.inverse_transform(dec)


class Model_score(nn.Module):
    def __init__(self, features):
        super(Model_score, self).__init__()
        # --- input use the feactures to define -- #
        self.nn1 = nn.Linear(features, 16).to(device)
        kaiming_uniform_(self.nn1.weight, nonlinearity='selu')
        self.nn2 = nn.Linear(16, 16).to(device)
        kaiming_uniform_(self.nn2.weight, nonlinearity='selu')
        self.nn3 = nn.Linear(16, 8).to(device)
        kaiming_uniform_(self.nn3.weight, nonlinearity='selu')

        self.drop = nn.Dropout(p=0.2).to(device)

    def forward(self, x):
        x = self.nn1(x)
        x = self.nn2(x)
        x = self.nn3(x)
        x = F.softmax(x, dim=1)

        return x


class Model(nn.Module):
    def __init__(self, features):
        super(Model, self).__init__()
        # --- input use the feactures to define -- #
        self.nn1 = nn.Linear(features, 1024).to(device)
        kaiming_uniform_(self.nn1.weight, nonlinearity='relu')
        self.nn2 = nn.Linear(1024, 1024).to(device)
        kaiming_uniform_(self.nn2.weight, nonlinearity='relu')
        self.nn3 = nn.Linear(1024, 512).to(device)
        kaiming_uniform_(self.nn3.weight, nonlinearity='relu')
        self.nn4 = nn.Linear(512, 8).to(device)
        kaiming_uniform_(self.nn4.weight, nonlinearity='relu')
        self.nn5 = nn.Linear(128, 64).to(device)
        kaiming_uniform_(self.nn5.weight, nonlinearity='relu')
        self.nn6 = nn.Linear(64, 32).to(device)
        kaiming_uniform_(self.nn6.weight, nonlinearity='relu')
        self.nn7 = nn.Linear(32, 16).to(device)
        kaiming_uniform_(self.nn7.weight, nonlinearity='relu')
        self.nn8 = nn.Linear(16, 8).to(device)
        kaiming_uniform_(self.nn8.weight, nonlinearity='relu')
        self.drop = nn.Dropout(p=0.8).to(device)

    def forward(self, x):
        x = self.nn1(x)
        x = self.drop(x)
        x = F.selu(x)
        x = self.nn2(x)
        x = self.drop(x)
        x = F.selu(x)

        x = self.nn3(x)
        x = self.drop(x)
        x = F.selu(x)
        x = self.nn4(x)
        # x = self.drop(x)
        x = F.softmax(x, dim=1)
        """
       
        
        
        x = self.nn5(x)
        # x = self.drop(x)
        x = F.relu(x)

        x = self.nn6(x)
        # x = self.drop(x)
        x = F.relu(x)

        x = self.nn7(x)
        # x = self.drop(x)
        x = F.relu(x)

        x = self.nn8(x)
        # x = self.drop(x)
        # x = F.relu(x)

        # x = self.nn9(x)
        # x = self.drop(x)
        # x = F.relu(x)

        # x = self.nn10(x)
        # x = self.drop(x)
        # x = F.log_softmax(x, dim=1)
        """
        return x


def train_model(x_train, model, epoches, model_name, x_test=None):
    # --- initialize lowest loss --- #
    lowest_loss = 100000

    # --- set up tensorboard --- #
    writer_train = SummaryWriter(comment='train')
    writer_test = SummaryWriter(comment='test')

    # --- Define the optimization --- #
    #criterion = nn.MSELoss(reduction='mean')
    criterion = nn.MSELoss()

    # --- define the optimizer --- #
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # --- Enumerate epochs --- #
    for epoch in range(epoches):

        total_train_loss = 0  # reset total train loss

        # --- Enumerate mini batches --- #
        for i, (inputs, targets) in enumerate(x_train):
            optimizer.zero_grad()  # clear the gradients
            yhat = model(inputs.to(device)).to(device)  # compute the model output
            yhat = yhat.squeeze()  # squeeze the data from (1xa) to (a)
            loss = criterion(yhat, targets.to(device)).to(device)  # calculate loss
            loss.backward()  # Credit assignment
            optimizer.step()  # update model weights

            total_train_loss += loss  # calculating the total train loss

            # --- print training process and result --- #
            print(f'\rEpoch {(epoch + 1):>5.5g}, train loss: {total_train_loss:.4e}  '
                  f'{str_load_process[int((i / len(x_train)) * 20)]}', end="")

        if x_test is not None:

            total_test_loss = 0
            test_predict = []
            test_answer = []

            for i, (inputs, targets) in enumerate(x_test):
                with torch.no_grad():
                    # --- Compute the model output --- #
                    yhat = model(inputs.to(device)).to(device)

                    yhat = yhat.squeeze()

                    # --- Calculate loss --- #
                    loss = criterion(yhat, targets.to(device)).to(device)

                    total_test_loss += loss

        # --- print the data to the terminal --- #
        if x_test is None:
            print(f'\rEpoch {(epoch + 1):>5.5g}, train loss: {total_train_loss:.4e}')
            print()
        else:
            print(f'\rEpoch {(epoch + 1):>5.5g}, train loss: {total_train_loss:.4e}, '
                  f'test loss: {total_test_loss:.4e} || '
                  f'avg train loss: {total_train_loss / len(x_train):.4e}, '
                  f'avg test loss: {total_test_loss / len(x_test):.4e}', end="")
            print()

        # --- save weight --- #
        if total_test_loss / len(x_test) < lowest_loss:
            lowest_loss = total_test_loss / len(x_test)
            torch.save(model, model_name, _use_new_zipfile_serialization=False)
            print('Model saved.')
            pass

        # --- write data to tensorboard --- #
        writer_train.add_scalar('total', total_train_loss, epoch, 1)
        writer_test.add_scalar('total', total_test_loss, epoch, 1)

        writer_train.add_scalar('avg', total_train_loss / len(x_train), epoch, 1)
        writer_test.add_scalar('avg', total_test_loss / len(x_test), epoch, 1)


class DLProcess(Dataset):
    def __init__(self, value_data_input, value_data_output):
        self.torch_data_input = torch.tensor(value_data_input, dtype=torch.float32)
        self.torch_data_output = torch.tensor(value_data_output, dtype=torch.float32)

        self.len = value_data_input.shape[0]

    def __getitem__(self, idx):
        return self.torch_data_input[idx], self.torch_data_output[idx]

    def __len__(self):
        return self.len


if __name__ == '__main__':

    batch_size = 32
    word_list = pd.read_pickle('hsv/hsv_list_spe.pkl')
    word_list = word_list.loc[word_list.saturation >= 0.6]
    word_list = word_list.loc[word_list.intensity >= 10]
    word_list_valid = word_list.words

    # DS = pd.read_pickle('result_all.pkl')[:500000]
    DS = pd.read_pickle('Dataset/DS_train.pkl')[:50000]
    msk = np.random.rand(len(DS)) <= 0.8

    train_df = DS[msk].reset_index(drop=True)
    test_df = DS[~msk].reset_index(drop=True)

    # --- training data --- #
    # train_input = pd.DataFrame(train_df.score.to_list())
    train_input = train_df.lemmas
    train_output = train_df.emotion

    # --- test data --- #
    # test_input = pd.DataFrame(test_df.score.to_list())
    test_input = test_df.lemmas
    test_output = test_df.emotion

    BOW = CountVectorizer()
    BOW.fit(word_list.words)

    train_input = BOW.transform(train_input)
    test_input = BOW.transform(test_input)

    label_encoder = LabelEncoder()
    label_encoder.fit(train_output)

    train_output = label_encode(label_encoder, train_output)
    test_output = label_encode(label_encoder, test_output)



    TORCH_DS_TRAIN = DLProcess(train_input.toarray(), train_output)
    TORCH_DS_TEST = DLProcess(test_input.toarray() , test_output)

    DL_DS_TRAIN = DataLoader(TORCH_DS_TRAIN, shuffle=True, batch_size=batch_size, drop_last=True)
    DL_DS_TEST = DataLoader(TORCH_DS_TEST, shuffle=True, batch_size=batch_size, drop_last=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'The program will be applied by', {device})

    num_inputs = train_input.shape[1]

    MD = Model(num_inputs)
    print(MD)

    train_model(x_train=DL_DS_TRAIN, x_test=DL_DS_TEST, model=MD, epoches=30000)

    pass
