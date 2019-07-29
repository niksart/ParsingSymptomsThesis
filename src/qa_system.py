from deeppavlov import build_model, configs
import commons as COM
import json

# Question tags
GENERAL = "general"
BODY_PARTS = "body_parts"
WHEN_STARTED = "when_started"

# Placeholders in questions
BODY_PART = "BODY_PART"



class QASystem:


  def __init__(self, model_type, model=None):
    if model_type == "bert":
      conf = configs.squad.squad_bert
    elif model_type == "rnet":
      conf = configs.squad.squad
    else:
      raise Exception("Model type should be 'bert' or 'rnet'.")
    
    if model == None:
      self.model = build_model(conf, download=True)
    elif str(type(model)) == "<class 'deeppavlov.core.common.chainer.Chainer'>":
      self.model = model
    
    # get the questions
    with open(COM.JSON_FILE_QUESTIONS_QA, "r") as qa_file:
      text = qa_file.read()
      self.__questions = json.loads(text)
  
  
  """
  Given a sentence object of class PreprocessedSentence, get the answers to the questions
  """
  def get_answers_given_sentence(self, sentence):
    res = dict()
    for question_obj in self.__questions:
      if BODY_PARTS in question_obj["tags"]:
        if sentence.has_body_parts():
          body_parts = sentence.get_body_parts()
          for body_part in body_parts:
            question_text = self.get_question_text(question_obj, body_part=body_part)
            answer = self.model([sentence.text], [question_text])[0][0]
            res[question_obj["question_type"] + "_" + str(body_part.id_bp)] = answer
      elif GENERAL in question_obj["tags"]:
        question_text = self.get_question_text(question_obj)
        answer = self.model([sentence.text], [question_text])[0][0]
        res[question_obj["question_type"]] = answer
      
      print(question_text)

    return res


  def get_question_text(self, question_obj, body_part=None):
      if len(question_obj["question"]) == 1:
        return question_obj["question"][0]
      else:
        ret = ""
        for token in question_obj["question"]:
          if token == BODY_PART:
            ret += body_part.exact_word
          else:
            ret += token
        return ret
