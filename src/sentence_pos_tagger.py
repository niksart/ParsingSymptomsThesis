import stanfordnlp
import commons as COM
import os

class SentencePOSTagger:
  
  possessive_verbs = ["have"]
  
  
  def __init__(self):
    if not os.path.isdir(COM.STANFORD_NLP_RESOURCES_PATH):
      stanfordnlp.download('en', force = True, resource_dir = COM.STANFORD_NLP_RESOURCES_PATH)
    
    self.nlp = stanfordnlp.Pipeline(models_dir = COM.STANFORD_NLP_RESOURCES_PATH)
  
  
  """
  Given one sentence in text, returns a Sentence object
  """
  def parse_sentence(self, sentence_text):
    doc = self.nlp(sentence_text)
    return doc.sentences[0]
  
  """
  def interpret_possession(self, sentence):
    deps_possession = ["root", "conj"]
    
    for token in sentence.tokens:
      for word in token.words:
        if word.dependency_relation in deps_possession \
        and word.lemma in possessive_verbs:
  """
  
  
  """
  Given a Token, it returns if the token is the negation adverb
  """
  def is_negation(self, token):
    if len(token.words) != 1:
      return False
    
    lemma_is_not = token.words[0].lemma == "not"
    dependency_is_advmod = token.words[0].dependency_relation == "advmod"
    
    return lemma_is_not and dependency_is_advmod