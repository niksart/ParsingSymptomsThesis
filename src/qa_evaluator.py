import commons as COM
import json

class QAEvaluator:
  
  def __init__(self, type_qa_system):
    if type_qa_system not in COM.QA_SYSTEM_TYPES:
      raise Exception("Invalid qa system type.")
    
    if type_qa_system == "bert":
      with open(COM.JSON_FILE_QASYSTEM_BERT_EVAL, "r") as f:
        self.__results = json.load(f)
    
    if type_qa_system == "rnet":
      with open(COM.JSON_FILE_QASYSTEM_RNET_EVAL, "r") as f:
        self.__results = json.load(f)
    
  
  def evaluate(self):
    printable_results = dict()
    l = []
    
    correct_tokens_total = 0
    wrong_tokens_total = 0
    non_sense_tokens_total = 0
    redundant_tokens_total = 0
    missing_tokens_total = 0
    
    for result in self.__results:
      correct_tokens_total += result["correct_tokens"]
      wrong_tokens_total += result["wrong_tokens"]
      non_sense_tokens_total += result["non_sense_tokens"]
      redundant_tokens_total += result["redundant_tokens"]
      missing_tokens_total += result["missing_tokens"]
      
      try:
        if result["interesting"] == "yes":
          l.append(result)
      except KeyError:
        continue
      
    total_tokens = correct_tokens_total + wrong_tokens_total + non_sense_tokens_total + redundant_tokens_total
    
    printable_results["correct_tokens"] = correct_tokens_total
    printable_results["wrong_tokens"] = wrong_tokens_total
    printable_results["non_sense_tokens"] = non_sense_tokens_total
    printable_results["redundant_tokens"] = redundant_tokens_total
    printable_results["missing_tokens"] = missing_tokens_total
    printable_results["interesting_results"] = l
    printable_results["score"] = correct_tokens_total * 2 + \
                                 wrong_tokens_total * -2 + \
                                 non_sense_tokens_total * -1 + \
                                 redundant_tokens_total * 0 + \
                                 missing_tokens_total * -2
    printable_results["accuracy"] = (correct_tokens_total + redundant_tokens_total) / total_tokens
    printable_results["missing_tokens/correct_tokens"] = missing_tokens_total / correct_tokens_total
    
    return printable_results
  
      