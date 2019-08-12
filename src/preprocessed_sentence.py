import commons as COM
import pandas as pd
from body_part import BodyPart


class PreprocessedSentence:
  
  
  def __init__(self, sentence_text, tokenizer):
    bpt = BodyPartTagger(tokenizer)
    self.text = sentence_text
    self.__body_parts = bpt.get_body_parts_in_sentence(sentence_text)
    
  
  def has_body_parts(self):
    return len(self.__body_parts) != 0
  
  
  def get_body_parts(self):
    return self.__body_parts
  
  
  
class BodyPartTagger:
  
  
  def __init__(self, tokenizer):
    self.__tokenizer = tokenizer
    
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
    tokens = self.__tokenizer.tokenize(sentence)
    
    found_body_parts = []
    for body_part in self.body_parts:
      for token in tokens:
        if token.text in body_part.names:
          body_part.exact_word = token.text
          found_body_parts.append(body_part)

    return found_body_parts
  
