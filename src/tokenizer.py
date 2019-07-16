import stanfordnlp as snlp
import commons as COM

class Tokenizer:
  
  def __init__(self):
    self.nlp = snlp.Pipeline(models_dir=COM.STANFORD_NLP_RESOURCES_PATH,
                             processors="tokenize")
  
  
  """
  Tokenize a sentence, assume there is only 1 sentence
  """
  def tokenize(self, sentence):
    tokens = self.nlp(sentence).sentences[0].tokens
    words = list()
    for token in tokens:
      for word in token.words:
        words.append(word)
    
    return words
  