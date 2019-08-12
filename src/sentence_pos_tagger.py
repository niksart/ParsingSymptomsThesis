import stanfordnlp
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
import commons as COM
import os

class SentencePOSTagger:
  
  
  def __init__(self):
    if not os.path.isdir(COM.STANFORD_NLP_RESOURCES_PATH):
      stanfordnlp.download('en', force = True, resource_dir = COM.STANFORD_NLP_RESOURCES_PATH)
    
    self.nlp = stanfordnlp.Pipeline(models_dir = COM.STANFORD_NLP_RESOURCES_PATH)
    
    glove_100000_word_list_df = pd.read_csv(COM.CSV_WORD_LIST_GLOVE_100000, header=None)
    self.__stemmer = SnowballStemmer("english")
    self.__list_glove_words = glove_100000_word_list_df[0].tolist()
  
  
  """
  Given one sentence in text, returns a Sentence object
  """
  def parse_sentence(self, sentence_text):
    doc = self.nlp(sentence_text)
    return doc.sentences[0]
  
  
  """
  Keep only the words whose type is in this list (referring to UPOS classification):
  
    - NOUN nouns
    - ADJ adjectives
    - CCONJ coordinating conjunctions
    - PUNCT punctuation
    - VERB verbs
    - AUX auxiliaries
  """
  def filter_unuseful_words(self, 
                            sentence_text,
                            admitted_types = ["NOUN", "ADJ", "CCONJ", "PUNCT", "VERB", "AUX"]):
    ambiguous_words = ["back"]
    
    if sentence_text == "":
      return ""
    
    sentence = self.parse_sentence(sentence_text)
    
    res = ""
    for word in sentence.words:
      if word.upos in admitted_types or word.text in ambiguous_words:
        res += word.text
        if word != sentence.words[-1]:
          res += " "
    
    return res
  
  
  """
  Given a string containing a sentence,
  returns a string (the stemmed sentence)
  """
  def stem_sentence(self, sentence_text):
    if sentence_text == "":
      return sentence_text
    
    sentence = self.parse_sentence(sentence_text)
    
    res = ""
    for word in sentence.words:
      stemmed_word = self.__stemmer.stem(word.text)
      
      if stemmed_word in self.__list_glove_words:
        res += stemmed_word
      else:
        res += word.text
      
      if word != sentence.words[-1]:
        res += " "
    
    return res
  
  