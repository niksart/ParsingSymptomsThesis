import commons as COM
import pandas as pd
import os


class MahalanobisTagger:
  
  
  def __init__(self,
               dimension_glove_vec = COM.POSSIBLE_GLOVE_DIMENSIONS[COM.DEFAULT_GLOVE_DIMENSION]):
    
    if dimension_glove_vec not in COM.POSSIBLE_GLOVE_DIMENSIONS:
      raise Exception("Dimension_glove_vector should be 50 or 100 or 200 or 300")
      
    self.dimension_glove_vec = dimension_glove_vec
    
    if not os.path.isfile(COM.CSV_ST_MEDICAL_WORDS_GLOVE_PATH + 
                          str(dimension_glove_vec) + "d.csv"):
      self.__save_csv_medical_vectors()
    
    
  def __save_csv_medical_vectors(self):
    medical_terms = pd.read_csv(COM.CSV_ST_MEDICAL_TERMS_FILE)
    
    medical_terms_list = medical_terms.iloc[:,0].tolist()
    
    word_vectors_df = pd.read_csv(COM.TXT_GLOVE_PATH + \
                                  "glove.6B." + str(self.dimension_glove_vec) + "d.txt",
                                  sep=" ",
                                  engine="python",
                                  nrows=COM.MAX_NUM_IMPORTED_GLOVE_VECTORS,
                                  quoting=3,
                                  header=None)
    
    medical_vec = word_vectors_df[word_vectors_df[0].isin(medical_terms_list)]
    medical_vec.to_csv(COM.CSV_ST_MEDICAL_WORDS_GLOVE_PATH + 
                       str(self.dimension_glove_vec) + "d.csv")
    
    
  