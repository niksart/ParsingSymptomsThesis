from body_part import BodyPart
from sentence_pos_tagger import SentencePOSTagger
import re


# QUESTION TYPES
SYMPTOMS = "SYMPTOMS"
PROBLEM_BODY_PART = "PROBLEM_BODY_PART"
WHEN_STARTED = "WHEN_STARTED"


class AnswerInterpreter:
  
  
  def __init__(self):
    self.__spt = SentencePOSTagger()  
  
  """
  answers is a dictionary built in this way:
    answers['type_question'] = 'answer text...'
  """
  def get_tokens_given_answers(self, answers):
    tokens_symptoms = self.get_tokens_given_text(answers[SYMPTOMS])
    
    tokens_bp = []
    for key in answers.keys():
      if self.__is_problem_body_part_key(key):
        id_bp = self.__get_id_body_part_by_key(key)
        bp = BodyPart(id_bp)
        tokens = self.get_tokens_given_text(answers[key] + " " + bp.names[0])
        tokens_bp.append((bp, tokens))
    
    return [tokens_symptoms, tokens_bp]
  
  
  """
  Given an answer_text
  returns a list of strings, that are the tokens that have to be vectorified
  and searched in the symptom tree
  
  If split_on_conj is True, the original text is split on the English coordinating
  conjunctions
  """
  def get_tokens_given_text(self, answer_text, split_on_conj=True):
    # preprocess answer text
    answer_text = self.preprocess_answer_text(answer_text)
    
    # split on the conjunctions (and, but, or, ...)
    if split_on_conj:
      answer_tokens = self.split_sentence_on_conjunction(answer_text)
    
    return answer_tokens
  
  
  """
  Given an answer text
  1- delete unuseful words
  2- pass it to the stemmer
  
  Both options can be disabled passing False to the arguments
  """
  def preprocess_answer_text(self, 
                             answer_text,
                             filter_unuseful_words=True,
                             stemming=True):
    # 1- delete unuseful words
    if filter_unuseful_words:
      answer_text = self.__spt.filter_unuseful_words(answer_text)
    # ---
    # 2- pass it to the stemmer
    if stemming:
      answer_text = self.__spt.stem_sentence(answer_text)
    # ---
    
    return answer_text
  
  
  """
  Given a sentence text,
  returns a list of strings (created splitting on conjunctions)
  """
  def split_sentence_on_conjunction(self, sentence_text):
    conjunctions = [",", "for", "and", "nor", "but", "or", "yet", "so"]
    conj_regex = ""
    for conj in conjunctions:
      conj_regex += conj
      if conj != conjunctions[-1]:
        conj_regex += " | "
      
    # split on coordinating conjunctions
    sentence_text = re.split(conj_regex, sentence_text)
    # ---
    
    return sentence_text
    
  
  
  def __is_problem_body_part_key(self, key):
    return PROBLEM_BODY_PART in key
  
  
  """
  example: eye has id 5
  the key of the question about the eye will be
  
  'PROBLEM_BODY_PART_5'
  
  Given the key, this functions returns an integer, the id of the body part (5)
  """
  def __get_id_body_part_by_key(self, key):
    if self.__is_problem_body_part_key(key):
      return int(key.split("_")[-1])
    else:
      raise Exception("The answer type is not " + PROBLEM_BODY_PART)
  