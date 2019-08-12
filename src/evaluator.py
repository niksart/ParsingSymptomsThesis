import commons as COM
import json
import pandas as pd
from symptom_tree import SymptomTree



class Evaluator:
  
  
  def __init__(self,
               qa_system,
               internal_representation,
               search_for_body_parts,
               pruning,
               min_similarity,
               filter_unuseful_words,
               glove_dimension=None):
    
    # qa_system must be in the list
    if qa_system not in COM.QA_SYSTEM_TYPES:
      raise Exception("Model type should be 'bert' or 'rnet'.")
    self.qa_system = qa_system
    
    # internal representation must be in the list
    if internal_representation not in COM.ADMITTED_REPRESENTATIONS:
      raise Exception("A proper internal representation should be specified.\n"
                      "Supported representations: " + ", ".join(COM.ADMITTED_REPRESENTATIONS))
    self.internal_representation = internal_representation
    
    if internal_representation == "glove":
      if glove_dimension not in COM.POSSIBLE_GLOVE_DIMENSIONS:
        raise Exception("Dimension_glove_vector should be 50 or 100 or 200 or 300")
      self.vector_dimension = glove_dimension
    elif internal_representation == "bert":
      self.vector_dimension = COM.DIMENSION_BERT_EMBEDDINGS
    
    if 0 <= min_similarity <= 1:
      if str(min_similarity) == "0.0":
        min_similarity = 0
    else:
      raise Exception("min_similarity must be a number >= 0 and <= 1")
    
    filename = COM.JSON_RESULTS_FOLDER + \
               qa_system + "-" + \
               internal_representation + \
               str(self.vector_dimension) + "-" + \
               ("search_bp" if search_for_body_parts else "no_search_bp") + "-" + \
               ("pruning" if pruning else "no_pruning") + "-" + \
               "min_sim" + str(min_similarity) + "-" + \
               ("filter_words" if filter_unuseful_words else "no_filter_words")
               
    extension = ".json"
    self.__filename_evaluated = filename + "-EVALUATED" + extension
    self.__filename_statistics = filename + "-STATISTICS" + extension
    filename += extension
    
    # Load json predictions
    try:
      with open(filename) as f:
        self.predictions = json.load(f)
    except FileNotFoundError:
      print("First compute the results with the Predictor, then evaluate them.")
    
    self.__st_light = SymptomTree()
  
  
  def evaluate(self):
    results = []
    
    for prediction in self.predictions:
      correct_cuis = 0
      correct_redundant_cuis = 0
      wrong_cuis = 0
      
      cui_predictions = prediction["cui_predictions"]
      real_cuis = prediction["real_cuis"]
      
      # List of subtrees: each subtree is a list of cuis
      real_cuis_expanded = self.__expand_real_cuis(real_cuis)
      subtrees_taken_yet = []
      
      # for each cui prediction
      for pred_cui in cui_predictions:
        # matching not found
        found_match = False
        
        # for each subtree in the list of subtrees
        for subtree in real_cuis_expanded:
          # if I've not found yet the matching
          if found_match == False:
            # if the cui is in this subtree
            if pred_cui in subtree:
              correct_cuis += 1
              # the subtree has to be removed because the associated symptom is gone
              real_cuis_expanded.remove(subtree)
              subtrees_taken_yet.append(subtree)
              found_match = True
        
        # provo a vedere se il cui previsto è in un sottoalbero già uscito
        # infatti se così fosse non è sbagliato, è solo una previsione ridondante (meno grave)
        for subtree_taken_yet in subtrees_taken_yet:
          if found_match == False:
            if pred_cui in subtree_taken_yet:
              correct_redundant_cuis += 1
              found_match = True
        
        # if found_match remains set to False => I haven't found any match => cui was wrong
        if found_match == False:
          wrong_cuis += 1
      
      missed_cuis = len(real_cuis) - correct_cuis
      real_cuis_num = len(real_cuis)
      cui_predictions_num = len(cui_predictions)
      
      prediction["correct_cuis"] = correct_cuis
      prediction["correct_redundant_cuis"] = correct_redundant_cuis
      prediction["wrong_cuis"] = wrong_cuis
      prediction["missed_cuis"] = missed_cuis
      prediction["real_cuis_num"] = real_cuis_num
      prediction["cui_predictions_num"] = cui_predictions_num
      prediction["accuracy"] = correct_cuis / real_cuis_num
      prediction["attempts"] = cui_predictions_num / real_cuis_num
      
      results.append((correct_cuis,
                      correct_redundant_cuis,
                      wrong_cuis,
                      missed_cuis,
                      real_cuis_num,
                      cui_predictions_num,
                      prediction["accuracy"],
                      prediction["attempts"]))
    
    # write down on results file
    with open(self.__filename_evaluated, "w") as f:
      json.dump(self.predictions, f, indent=2)
      print("Successfully saved results file.")
    
    mean_results_df = pd.DataFrame(results, columns=["correct_cuis",
                                                     "correct_redundant_cuis",
                                                     "wrong_cuis",
                                                     "missed_cuis",
                                                     "real_cuis_num",
                                                     "cui_predictions_num",
                                                     "accuracy",
                                                     "attempts"])
    
    return self.__get_printable_results(mean_results_df)
  
  
  def __get_printable_results(self, results_df):
    ret = dict()
    ret["number of non_redundant_correct_cuis"] = int(results_df["correct_cuis"].sum())
    ret["number of redundant_correct_cuis"] = int(results_df["correct_redundant_cuis"].sum())
    ret["number of wrong_cuis"] = int(results_df["wrong_cuis"].sum())
    ret["number of missed_cuis"] = int(results_df["missed_cuis"].sum())
    ret["number of real_cuis"] = int(results_df["real_cuis_num"].sum())
    ret["number of prediction_cuis"] = int(results_df["cui_predictions_num"].sum())
    ret["correct predictions / total predictions"] = (ret["number of non_redundant_correct_cuis"] + ret["number of redundant_correct_cuis"]) / ret["number of prediction_cuis"]
    ret["accuracy (correct predictions / real cuis number)"] = ret["number of non_redundant_correct_cuis"] / ret["number of real_cuis"]
    ret["attempts per real cui"] = ret["number of prediction_cuis"] / ret["number of real_cuis"]
    
    # write down on results file
    with open(self.__filename_statistics, "w") as f:
      json.dump(ret, f, indent=2)
      print("Successfully saved statistics file.")
    
    return ret
  
  
  def __expand_real_cuis(self, real_cuis):
    ret = []
    for cui in real_cuis:
      cuis = self.__st_light.get_cuis_subtree_given_cui_root(cui)
      ret.append(cuis)
    
    return ret
      
      