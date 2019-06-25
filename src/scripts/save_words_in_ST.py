import pandas as pd

chunk_size = 10 ** 4
dimension_vectors = [50, 100, 200, 300]
d = dimension_vectors[3]

csv_reader = pd.read_csv("../../txt/glove.6B."+str(d)+"d.txt", \
                        sep=" ", \
                        engine="python", \
                        chunksize=chunk_size, \
                        quoting=3, \
                        header=None, \
                        iterator=True)

df_words = pd.read_csv("../../csvs/words_of_ST.csv")
words_list = list(df_words.iloc[:,0].unique())

final_df = pd.DataFrame()

i=0
for df in csv_reader:
  print("chunk n "+str(i))
  i=i+1
  filtered_df = df[df[0].isin(words_list)]
  final_df = final_df.append(filtered_df)

final_df.to_csv("/home/niksart/HistoryTakingThesis/csvs/ST_word_glove_vectors_"+ str(d) +"d.csv", index=False, header=False)
