from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pandas as pd

train_df = pd.read_pickle('result.pkl')
train_output = train_df.output
train_predict = train_df.predict_emotion

train_acc = accuracy_score(y_true=train_output, y_pred=train_predict)
print(f'training accuracy: {train_acc:.2f}')
print(classification_report(y_true=train_output, y_pred=train_predict))
