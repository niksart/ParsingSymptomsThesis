import commons as COM
import json
from preprocessed_sentence import PreprocessedSentence
from qa_system import QASystem
from answers_interpreter import AnswersInterpreter
from sentence_pos_tagger import SentencePOSTagger
from vectorifier import Vectorifier
from symptom_tree import SymptomTree
from dataset_loader import DatasetLoader
from tokenizer import Tokenizer


class Predictor:
  
  """
  Predictor for predicting the dataset sentences
  
  qa_system_type
    can be or "rnet" or "bert"
  internal_representation_type
    can be or "glove" or "bert"
  min_similarity
    two internal representations are considered similar if the similarity > this value
  search_for_body_parts
    if true use the tokens originated from the body part search
  pruning
    prune the symptom tree in case of body parts
  filter_unuseful_words_from_tokens
    filter words
  number_glove_words
    default: 100000
  dimension_glove_vectors
    default: 100
  """
  def __init__(self,
               qa_system_type,
               internal_representation_type,
               min_similarity=0,
               search_for_body_parts=True,
               pruning=True,
               filter_unuseful_words_from_tokens=True,
               number_glove_words=COM.DEFAULT_NUM_IMPORTED_GLOVE_VECTORS,
               dimension_glove_vectors=COM.POSSIBLE_GLOVE_DIMENSIONS[COM.DEFAULT_GLOVE_DIMENSION]):
    # initialize QA system
    if qa_system_type in COM.QA_SYSTEM_TYPES:
      self.__qas = QASystem(qa_system_type)
    else:
      raise Exception("qa_system argument should be an element in this list :" + COM.QA_SYSTEM_TYPES)
    
    self.__search_for_body_parts = search_for_body_parts
    
    self.__pruning = pruning
    
    self.__filter_unuseful_words = filter_unuseful_words_from_tokens
    
    # initialize SentencePOSTagger
    self.__spt = SentencePOSTagger()
    
    #initialize Tokenizer
    self.__tokenizer = Tokenizer()
    
    # initialize answer interpreter
    self.__ai = AnswersInterpreter(spt=self.__spt)
    
    # initialize vectorifier
    self.__vect = Vectorifier(internal_representation_type,
                              number_glove_words,
                              dimension_glove_vectors,
                              self.__spt)
    
    # initialize symptom tree
    self.__st = SymptomTree(self.__vect)
    
    # initialize dataset loader
    self.__dl = DatasetLoader()
    
    if 0 <= min_similarity <= 1:
      if str(min_similarity) == "0.0":
        self.__min_similarity = 0
      else:
        self.__min_similarity = min_similarity
    else:
      raise Exception("min_similarity must be a number >= 0 and <= 1")
  
  
  """
  Predict and save the results on a file
  """
  def predict_testing(self):
    sentences_and_cuis = self.__dl.get_testing_sentences_and_cuis()
    
    sentences_and_tokens_for_pred = []
    results = []
    i = 1
    
    for text_sentence, sentence_cuis in sentences_and_cuis:
      print("Predicting n" + str(i) + ": \"" + text_sentence + "\"\n")
      cui_predictions, tokens_for_pred = self.predict(text_sentence)
      
      d1 = dict()
      d1["text_sentence"] = text_sentence
      d1["tokens_for_pred"] = tokens_for_pred
      d1["correct_tokens"] = 0
      d1["wrong_tokens"] = 0
      d1["non_sense_tokens"] = 0
      d1["redundant_tokens"] = 0
      d1["missing_tokens"] = 0
      sentences_and_tokens_for_pred.append(d1)
      
      d2 = dict()
      d2["text_sentence"] = text_sentence
      d2["cui_predictions"] = cui_predictions
      d2["real_cuis"] = sentence_cuis
      results.append(d2)
      
      i += 1
    
    if self.__qas.model_type == "rnet":
      # write file of sentences and tokens for QASystem evaluation
      with open(COM.JSON_FILE_QASYSTEM_RNET_EVAL_TO_TAG, 'w', encoding='utf-8') as f:
        json.dump(sentences_and_tokens_for_pred, f, ensure_ascii=False, indent=2)
      # ---
    
    if self.__qas.model_type == "bert":
      # write file of sentences and tokens for QASystem evaluation
      with open(COM.JSON_FILE_QASYSTEM_BERT_EVAL_TO_TAG, 'w', encoding='utf-8') as f:
        json.dump(sentences_and_tokens_for_pred, f, ensure_ascii=False, indent=2)
      # ---
    
    filename = COM.JSON_RESULTS_FOLDER + \
               self.__qas.model_type + "-" + \
               self.__vect.internal_representation + \
               str(self.__vect.d) + "-" + \
               ("search_bp" if self.__search_for_body_parts else "no_search_bp") + "-" + \
               ("pruning" if self.__pruning else "no_pruning") + "-" + \
               "min_sim" + str(self.__min_similarity) + "-" + \
               ("filter_words" if self.__filter_unuseful_words else "no_filter_words") + \
               ".json"
    
    #write file of results
    with open(filename, 'w', encoding='utf-8') as f:
      json.dump(results, f, ensure_ascii=False, indent=2)
    # ---
        
  
  """
  predictions: list of strings (cuis)
  text_tokens: list of tokens used for prediction (used for evaluation purposes of QASystem)
  """
  def predict(self, patient_sentence_text):
    
    # preprocess sentence (tag eventual body parts)
    preprocessed_sentence = PreprocessedSentence(patient_sentence_text, self.__tokenizer)
    
    # get answers given preprocessed sentence
    answers = self.__qas.get_answers_given_sentence(preprocessed_sentence)
    
    # get searching tokens of symptoms and of body parts
    tokens_symptoms, tokens_body_parts = self.__ai.get_tokens_given_answers(answers, self.__filter_unuseful_words)
    
    # get predictions CUI
    predictions = []
    text_tokens = []
    for token_symptom in tokens_symptoms:
      # returns a tuple (cui, similarity)
      d = self.__st.get_most_similar_cui_given_symptom_token(token_symptom,
                                                             min_similarity=self.__min_similarity)
      
      if d["found"] == "yes":
          predicted_cui = d["predicted_cui"]
      else:
          predicted_cui = None
      
      text_tokens.append(token_symptom)
      
      if predicted_cui != None:
        predictions.append(predicted_cui)
    
    # if is enabled the optimization for searching the body parts
    if self.__search_for_body_parts:
      for body_part, tokens_body_part in tokens_body_parts:
        for token_body_part in tokens_body_part:
          
          # if pruning, search only in the subtrees related to that body part
          # for evaluation purposes
          if self.__pruning:
            d = self.__st.get_most_similar_cui_given_body_part_token(token_body_part,
                                                                     body_part,
                                                                     min_similarity=self.__min_similarity)
          else:
            d = self.__st.get_most_similar_cui_given_symptom_token(token_body_part,
                                                                   min_similarity=self.__min_similarity)
            
          if d["found"] == "yes":
            predicted_cui = d["predicted_cui"]
          else:
            predicted_cui = None
          
          text_tokens.append(token_body_part)
          
          if predicted_cui != None:
            predictions.append(predicted_cui)
    
    return predictions, text_tokens
  
    