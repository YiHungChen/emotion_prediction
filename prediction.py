from FE_NN import DLProcess, DataLoader, device, Model, label_encode, label_decode
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import torch

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


if __name__ == '__main__':

    word_list = pd.read_pickle('hsv/hsv_list_spe.pkl')
    word_list = word_list.loc[word_list.saturation >= 0.4]
    word_list = word_list.loc[word_list.intensity >= 10]
    word_list_valid = word_list.words

    DS_train = pd.read_pickle('result_all.pkl')[:100000]
    DS_test = pd.read_pickle('Dataset/DS.pkl')

    DS_test = DS_test.loc[DS_test.identification == 'test']
    DS = pd.read_pickle('result_test.pkl')

    # --- training data --- #
    # upload_input = pd.DataFrame(DS.score.to_list())
    upload_input = DS_test.lemmas
    upload_output = [0] * len(upload_input)
    # upload_output = DS.predict_emotion

    label_encoder = LabelEncoder()
    label_encoder.fit(DS_train.output)

    BOW = CountVectorizer()
    BOW.fit(word_list.words)

    upload_input = BOW.transform(upload_input)

    # upload_output = label_encode(label_encoder, upload_output)

    TORCH_DS_UPLOAD = DLProcess(upload_input.toarray(), upload_output)
    DL_DS_UPLOAD = DataLoader(TORCH_DS_UPLOAD, shuffle=False, batch_size=1, drop_last=True)

    MD = torch.load('model/model_20211223-2310.pth')
    predictions = target_model(DL_DS_UPLOAD, MD)
    pred = label_decode(label_encoder, predictions)

    output = pd.DataFrame({'id': DS.id, 'emotion': pred})
    output.to_csv('results-Torch-NN-1223.csv', index=False)

