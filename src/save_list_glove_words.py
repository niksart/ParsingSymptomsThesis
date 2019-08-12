import commons as COM
import pandas as pd


def save_csv_glove_words():
    l = word_vectors_df[0]
    l.to_csv(COM.CSV_WORD_LIST_GLOVE_100000, index=None)


word_vectors_df = pd.read_csv(COM.TXT_GLOVE_PATH + \
                              "glove.6B." + str(50) + "d.txt",
                              sep=" ",
                              engine="python",
                              nrows=COM.DEFAULT_NUM_IMPORTED_GLOVE_VECTORS,
                              quoting=3,
                              header=None)

save_csv_glove_words()
