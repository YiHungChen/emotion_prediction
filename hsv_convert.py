import pandas as pd


def normalization(df):
    a = df.loc[:, 'anger':'trust']
    b = a.max(axis=1)
    c = a.min(axis=1)
    d = a.apply(lambda a: (a-c)/(b-c))
    df.loc[:, 'anger':'trust'] = d
    return df


word_list_df = pd.read_pickle('words_list/words_list.pkl')
word_list_df['intensity'] = word_list_df.loc[:, 'anger':'trust'].max(axis=1, numeric_only=True)
word_list_df = normalization(word_list_df)
word_list_df = word_list_df.dropna()
word_list_df['max'] = word_list_df.loc[:, 'anger':'trust'].max(axis=1, numeric_only=True)
word_list_df['3rd_max'] = word_list_df.loc[:, 'anger':'trust'].apply(lambda row: row.nlargest(3).values[-1], axis=1)
word_list_df['saturation'] = ((word_list_df['max'] - word_list_df['3rd_max']) / word_list_df['max'])

word_list_df.to_csv('hsv/hsv_list_spe.csv')
word_list_df.to_pickle('hsv/hsv_list_spe.pkl')
print(len(word_list_df))



