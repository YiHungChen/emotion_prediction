import pandas as pd

word_list_df = pd.read_pickle('Dataset/words_list.pkl')[50000:80000]
word_list_df['max'] = word_list_df.max(axis=1, numeric_only=True)
word_list_df['3rd_max'] = word_list_df.iloc[:, 2:-1].apply(lambda row: row.nlargest(2).values[-1], axis=1)
word_list_df['saturation'] = ((word_list_df['max'] - word_list_df['3rd_max']) / word_list_df['max'])
word_list_df['intensity'] = word_list_df.loc[:, 'anger':'trust'].mean(axis=1, numeric_only=True)
word_list_df.to_csv('words_list_3rd.csv')
word_list_df.to_pickle('Dataset/words_list_3rd.pkl')
print(len(word_list_df))
