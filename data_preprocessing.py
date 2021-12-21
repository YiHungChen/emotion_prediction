import pandas as pd
from lemmatization import lemmas


train_df = pd.read_pickle('Dataset/DS_train.pkl')
train_df['lemmas'] = lemmas(train_df.text)
print(train_df)

