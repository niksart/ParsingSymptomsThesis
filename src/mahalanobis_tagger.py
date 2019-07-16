import commons as COM
import pandas as pd
import numpy as np
from tokenizer import Tokenizer
from vectorifier import Vectorifier
import os
from scipy.spatial.distance import mahalanobis


class MahalanobisTagger:
  
  
  def __init__(self,
               dimension_glove_vec = COM.POSSIBLE_GLOVE_DIMENSIONS[COM.DEFAULT_GLOVE_DIMENSION],
               distance_measure_type = "mahalanobis"):
    
    if dimension_glove_vec not in COM.POSSIBLE_GLOVE_DIMENSIONS:
      raise Exception("Dimension_glove_vector should be 50 or 100 or 200 or 300")
      
    self.dimension_glove_vec = dimension_glove_vec
    
    if not os.path.isfile(COM.CSV_ST_MEDICAL_WORDS_GLOVE_PATH + 
                          str(dimension_glove_vec) + "d.csv"):
      self.__save_csv_medical_vectors()
    
    self.medical_vectors_df = pd.read_csv(COM.CSV_ST_MEDICAL_WORDS_GLOVE_PATH + 
                                          str(dimension_glove_vec) + "d.csv")
    
    if distance_measure_type == "mahalanobis":
      self.cov_matrix = self.medical_vectors_df.iloc[:,1:(dimension_glove_vec + 1)].cov()
    elif distance_measure_type == "euclidean":
      self.cov_matrix = np.eye(dimension_glove_vec)
    else:
      raise Exception("Distance measure type not supported.")
    
    
    self.centroid = self.medical_vectors_df.iloc[:,1:(dimension_glove_vec + 1)].mean()
    
    similarity_column = self.medical_vectors_df.apply(lambda row: mahalanobis(self.centroid, 
                                                                              row[1:(self.dimension_glove_vec + 1)],
                                                                              self.cov_matrix), 
                                                                  axis=1)
    self.medical_vectors_df = self.medical_vectors_df.assign(similarity=similarity_column)
    
    # Calculate mean and std of the distances from the centroid
    self.mean_distance = self.medical_vectors_df["similarity"].mean()
    self.std_distance = self.medical_vectors_df["similarity"].std()
    
    self.tokenizer = Tokenizer()
  
  
  def tag_sentence(self, sentence):
    vectorifier = Vectorifier("glove")
    tokens = self.tokenizer.tokenize(sentence)
    res=[]
    
    for token in tokens:
      token_text = token.text.lower()
      vector = vectorifier.vectorize_word(token_text)
      
      #se la parola non e' stata trovata
      if not vector.empty:
        maha_distance = mahalanobis(vector, self.centroid, self.cov_matrix)
      
        if (self.mean_distance - self.std_distance) < maha_distance < (self.mean_distance + self.std_distance):
          res.append("yes")
        else:
          res.append("no")
      else:
        res.append("no")
    
    return res
  
  
  def __save_csv_medical_vectors(self):
    medical_terms = pd.read_csv(COM.CSV_ST_MEDICAL_TERMS_FILE)
    
    word_vectors_df = pd.read_csv(COM.TXT_GLOVE_PATH + \
                                  "glove.6B." + str(self.dimension_glove_vec) + "d.txt",
                                  sep=" ",
                                  engine="python",
                                  nrows=COM.MAX_NUM_IMPORTED_GLOVE_VECTORS,
                                  quoting=3,
                                  header=None)
    
    medical_terms_list = medical_terms.iloc[:,0].tolist()
    
    medical_vec = word_vectors_df[word_vectors_df[0].isin(medical_terms_list)]
    medical_vec.to_csv(COM.CSV_ST_MEDICAL_WORDS_GLOVE_PATH + 
                       str(self.dimension_glove_vec) + "d.csv",
                       index=None)
    