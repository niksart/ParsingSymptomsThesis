import commons as COM
import pandas as pd
from tokenizer import Tokenizer


class Sentence:
  
  
  def __init__(self, sentence_text):
    bpt = BodyPartTagger()
    self.text = sentence_text
    self.__body_parts = bpt.get_body_parts_in_sentence(sentence_text)
    
  
  def has_body_parts(self):
    return len(self.__body_parts) != 0
  
  
  def get_body_parts(self):
    return self.__body_parts
  
  
  
class BodyPartTagger:
  
  
  def __init__(self):
    #carica file e lista di body parts
    self.body_parts = []
    body_parts_df = pd.read_csv(COM.CSV_BODY_PARTS_FILE)
    for _, row in body_parts_df.iterrows():
      bp = BodyPart(row["ID"],
                    row["NAME"],
                    row["REF_TO"],
                    row["IS_EXTERNAL"],
                    row["NOT_SPECIFICALLY_LOCATED"],
                    row["ONE_WORD"])
      self.body_parts.append(bp)
  
  
  def get_body_parts_in_sentence(self, sentence):
    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize(sentence)
    
    found_body_parts = []
    for body_part in self.body_parts:
      for token in tokens:
        if token.text in body_part.names:
          body_part.exact_word = token.text
          found_body_parts.append(body_part)
    
    return found_body_parts
  
  

"""
  Representation of a body part.
  
  id_bp: univoque id
  names: list of names of that bp
  ref_to: this bp is part of another bigger bp (pointed by ref_to)
  is_external: True if the bp is external
  not_specifically_located: True if the bp is not specifically located in a part of the body (e.g. for blood)
  one_word: True if all the names of the bp are made of one word
"""
class BodyPart:
  
  
  def __init__(self, id_bp, names_text, ref_to, is_external, not_specifically_located, one_word):
    self.id_bp = id_bp
    self.names = self.__get_list_of_names(names_text)
    self.ref_to = ref_to
    self.is_external = False if is_external == 0 else True
    self.not_specifically_located = False if not_specifically_located == 0 else True
    self.one_word = False if one_word == 0 else True 
    
  
  def __get_list_of_names(self, names_text):
    return names_text.split("/")
    