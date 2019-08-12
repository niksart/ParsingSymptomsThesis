import pandas as pd
import commons as COM
import itertools
from sentence_pos_tagger import SentencePOSTagger
from bert_serving.client import BertClient


class Vectorifier:

  
  """
  Vectorifier converts a sentence into an internal representation (glove).
  
  internal_representation: can be "glove" or "bert"
  
  GloVe parameters --
    number_glove_words: take the first n words from GloVe dictionary (they are ordered by frequency)
    dimension_glove_vectors: can be 50, 100, 200, 300
    
  spt: SentencePOSTagger instance for optimization. If not passed, initialize a new instance
  """
  def __init__(self,
               internal_representation,
               number_glove_words = COM.DEFAULT_NUM_IMPORTED_GLOVE_VECTORS,
               dimension_glove_vectors = COM.POSSIBLE_GLOVE_DIMENSIONS[COM.DEFAULT_GLOVE_DIMENSION],
               spt=None):
    
    if internal_representation not in COM.ADMITTED_REPRESENTATIONS:
      raise Exception("A proper internal representation should be specified.\n"
                      "Supported representations: " + ", ".join(COM.ADMITTED_REPRESENTATIONS))
    
    self.internal_representation = internal_representation
    
    if internal_representation == "glove":
      if dimension_glove_vectors not in COM.POSSIBLE_GLOVE_DIMENSIONS:
        raise Exception("Dimension_glove_vector should be 50 or 100 or 200 or 300")
      
      if number_glove_words > COM.MAX_NUM_IMPORTED_GLOVE_VECTORS:
        raise Exception("The maximum number of glove words is " + str(COM.MAX_NUM_IMPORTED_GLOVE_VECTORS))
      
      self.d = dimension_glove_vectors
      
      self.__word_vectors_df = pd.read_csv(COM.TXT_GLOVE_PATH + \
                                           "glove.6B." + str(self.d) + "d.txt",
                                           sep=" ",
                                           engine="python",
                                           nrows=number_glove_words,
                                           quoting=3,
                                           header=None)
    elif internal_representation == "bert":
      self.d = COM.DIMENSION_BERT_EMBEDDINGS
      # Initialize a new BertClient
      self.__bc = BertClient(ip=COM.IP_BERT_SERVER, output_fmt="list")
      # ---
    
    if spt == None:
      self.__spt = SentencePOSTagger()
    else:
      self.__spt = spt
    
  
  
  """
  Given a sentence and a knowledge base (mapping word => vector),
  calculate the mean vector between the vectorified words

  If the concept name contains a word that is not in the dictionary, return an empty DataFrame
  
  kb: column 0 => word
      columns [1, dim + 1] => values of the vector
  """
  def __vectorize(self, sentence, kb=None):
    if kb is None:
      # BERT
      if sentence.strip() != "":
        vector_list = self.__bc.encode([sentence])
        return pd.Series(vector_list[0])
      else:
        return pd.Series()
    else:
      # GLOVE
      tokens = sentence.split()
      n_tokens = 0
      
      final_df = pd.DataFrame()
      
      for token in tokens:
        token_df = kb[kb[0].isin([token])]  
        if not token_df.empty:
          n_tokens = n_tokens + 1
        else:
          return pd.DataFrame() 
        final_df = final_df.append(token_df)
      
      if not final_df.empty:
        final_df = final_df.iloc[:,1:(self.d + 1)]
        return final_df.sum(axis=0) / n_tokens
      else:
        return pd.DataFrame()
    
  
  def vectorize_word(self, word, stemming=True):
    if stemming:
      word = self.__spt.stem_sentence(word)
    
    if self.internal_representation == "glove":
      vector = self.__vectorize(word, self.__word_vectors_df)
    elif self.internal_representation == "bert":
      vector = self.__vectorize(word)
    
    return vector
  
  
  """
  Given a concept name returns a generic vector representing it
  """
  def vectorize_concept_name(self,
                             concept_name,
                             stemming=True):
    # concept_name is stripped yet by get_concept_name
    
    print("Vectorizing concept name: " + concept_name)
    
    # stem the concept name
    if stemming:
      concept_name = self.__spt.stem_sentence(concept_name)
    # ---
    
    if self.internal_representation == "glove":
      return self.__vectorize(concept_name, self.__word_vectors_df)
    elif self.internal_representation == "bert":
      return self.__vectorize(concept_name)
  
  
  """
  Given a sentence,
    return for each subsentence a tuple (subsentence, subsentence vectorified)
    with the right internal representation
  """
  def get_subsentences_and_vectors(self, sentence):
    words_set = set(sentence.split(" "))
    subsentence_list = self.__subsentences(words_set)
    
    if self.internal_representation == "glove":
      subsentence_vectorified_list = [self.__vectorize(s, self.__word_vectors_df) for s in subsentence_list]
    elif self.internal_representation == "bert":
      subsentence_vectorified_list = [self.__vectorize(s) for s in subsentence_list]
    
    #delete subsentences that produce as vector an empty dataframe
    ret = list(zip(subsentence_list, subsentence_vectorified_list))
    ret1 = []
    for t in ret:
      (_, vect) = t
      if not vect.empty:
        ret1.append(t)

    return ret1
  
  
  def __subsentences(self, word_set):
    subsets_of_words = self.__subsets(word_set)
    return list(map(self.__stringify_tuple, subsets_of_words))
  
  
  # S is the set, m the number 
  def __subsets(self, S):
    final_set = set()
    for m in range(1, len(S) + 1):
      s = set(itertools.combinations(S, m))
      final_set = final_set.union(s)
    return final_set
  
  
  # given a tuple of strings, return a mono string
  def __stringify_tuple(self, t):
    s = ""
    for string in t:
        s = s + string + " "
    return s
  
  
