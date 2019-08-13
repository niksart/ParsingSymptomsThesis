charts_type = 2
missed_bool = False
# ----------------------------------------------------------------------------

import os, glob
import ntpath
import json
import numpy as np
import matplotlib.pyplot as plt

RESULT_JSON_FOLDER = "/home/niksart/ParsingSymptomsThesis/data/json/results/"

result_paths = [f for f in glob.glob(RESULT_JSON_FOLDER + "*STATISTICS.json") if os.path.isfile(f)]

results_json = dict()
for result_path in result_paths:
  result_filename = ntpath.basename(result_path)
  
  with open(result_path, "r") as f:
    result_json = json.load(f)
  
  results_json[result_filename] = result_json
  

charts_specs = [
    # 0 - files for comparing different types of embeddings
    (("bert-glove50-search_bp-pruning-min_sim0-filter_words-STATISTICS.json",
      "bert-glove100-search_bp-pruning-min_sim0-filter_words-STATISTICS.json",
      "bert-glove200-search_bp-pruning-min_sim0-filter_words-STATISTICS.json",
      "bert-glove300-search_bp-pruning-min_sim0-filter_words-STATISTICS.json",
      "bert-bert768-search_bp-pruning-min_sim0-filter_words-STATISTICS.json"),
      ("glove 50", "glove 100", "glove 200", "glove 300", "bert 768"),
      ("#00b809", "#75ff7c", "#d90000", 0.4),
      ("n° of predictions", "Composition of predictions by embeddings type")),
    
    # 1- files for different levels of min similarity thresholds
    (("bert-glove200-search_bp-pruning-min_sim0-filter_words-STATISTICS.json",
      "bert-glove200-search_bp-pruning-min_sim0.7-filter_words-STATISTICS.json",
      "bert-glove200-search_bp-pruning-min_sim0.8-filter_words-STATISTICS.json",
      "bert-glove200-search_bp-pruning-min_sim0.9-filter_words-STATISTICS.json",
      "bert-glove200-search_bp-pruning-min_sim0.95-filter_words-STATISTICS.json"),
      ("min. s. 0.0", "min. s. 0.70", "min. s. 0.80", "min. s. 0.90", "min. s. 0.95"),
      ("#00b809", "#75ff7c", "#d90000", 0.4),
      ("n° of predictions", "Composition of predictions by min. similarity")),
    
    # 2 - files for pruning / no pruning
    (("bert-glove200-search_bp-pruning-min_sim0-filter_words-STATISTICS.json", 
      "bert-glove200-search_bp-no_pruning-min_sim0-filter_words-STATISTICS.json"), 
     ("pruning", "no pruning"),
     ("#00b809", "#75ff7c", "#d90000", 0.1),
     ("n° of predictions", "Comparison of predictions\nbetween pruning and no pruning")),
    
    # 3 - files for filtering / no filtering
    (("bert-glove200-search_bp-pruning-min_sim0-filter_words-STATISTICS.json", 
      "bert-glove200-search_bp-pruning-min_sim0-no_filter_words-STATISTICS.json"),
     ("filter", "no filter"),
     ("#00b809", "#75ff7c", "#d90000", 0.1),
     ("n° of predictions", "Comparison of predictions\nbetween filtering and no filtering unuseful words")),    

    # 4 - files for search / no search body parts
    (("bert-glove200-search_bp-pruning-min_sim0-filter_words-STATISTICS.json", 
      "bert-glove200-no_search_bp-pruning-min_sim0-filter_words-STATISTICS.json"), 
     ("search", "no search"),
     ("#00b809", "#75ff7c", "#d90000", 0.1),
     ("n° of predictions", "Comparison of predictions\nbetween searching and not searching for body parts"))
]

N = len(charts_specs[charts_type][0])

correct_no_red_list = []
correct_red_list = []
wrong_list = []
missed_list = []

for filename in charts_specs[charts_type][0]:
  correct_no_red_list.append(results_json[filename]["number of non_redundant_correct_cuis"])
  correct_red_list.append(results_json[filename]["number of redundant_correct_cuis"])
  wrong_list.append(results_json[filename]["number of wrong_cuis"])
  missed_list.append(results_json[filename]["number of missed_cuis"])

correct_no_red = tuple(correct_no_red_list)
correct_red = tuple(correct_red_list)
correct_sum = [sum(x) for x in zip(correct_no_red, correct_red)]
wrong = tuple(wrong_list)
missed = tuple(missed_list)

if missed_bool == False:
  ind = np.arange(N)    # the x locations for the groups
  if N==2:
    ind=[0.2, 0.4]
  width = charts_specs[charts_type][2][3]      # the width of the bars: can also be len(x) sequence
  
  color1 = charts_specs[charts_type][2][0]
  p1 = plt.bar(ind, correct_no_red, width, color=color1)
  color2 = charts_specs[charts_type][2][1]
  p2 = plt.bar(ind, correct_red, width, bottom=correct_no_red, color=color2)
  color3 = charts_specs[charts_type][2][2]
  p3 = plt.bar(ind, wrong, width, bottom=correct_sum, color=color3)
  
  plt.ylabel(charts_specs[charts_type][3][0])
  plt.title(charts_specs[charts_type][3][1])
  plt.xticks(ind, charts_specs[charts_type][1])
  plt.yticks(np.arange(0, 850, 100))
  plt.legend((p1[0], p2[0], p3[0]), 
             ("Correct\n(non redundant)", "Correct\n(redundant)", "Wrong"),
             bbox_to_anchor=(1.4, 1.05))
  
  y_line_annotation = 467
  ax = plt.gca()
  
  ax.axhline(y=y_line_annotation, linestyle = "dashed", color="#000000", alpha=0.8)
  
  if N!=2:
    ax.text(y=y_line_annotation-20, x=4.45, s="n° of total symptoms\n       in sentences")
    
  
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  if N==2:
    ax.set_xlim(0,0.6)
    ax.text(y=y_line_annotation-20, x=0.60, s="n° of total symptoms\n       in sentences")
  
  plt.show()
else:
  # missed graph
  plt.rcdefaults()
  fig, ax = plt.subplots()
  
  # Example data
  features = charts_specs[charts_type][1]
  
  if N!=2:
    y_pos = np.arange(N)
  else:
    y_pos = [0.2, 0.35]
    
  performance = missed
  
  if N==2:
    width = 0.06
  else:
    width = 0.5
  
  ax.barh(y_pos, performance, width, align='center', color="#d90000")
  
  ax.set_yticks(y_pos)
  ax.set_yticklabels(features)
  ax.invert_yaxis()  # labels read top-to-bottom
  ax.set_xlabel('Missed symptoms')
  ax.set_title('Symptoms missings')
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  ax.set_xlim(0,500)
  ax.axvline(x=467, linestyle = "dashed", color="#000000", alpha=0.8)
  if N!=2:
    ax.text(x=475, y=-0.2, s="n° of total symptoms\n       in sentences", fontsize="small")
  else:
    ax.text(x=475, y=0.425, s="n° of total symptoms\n       in sentences", fontsize="small")
  
  if N==2:
    ax.set_ylim(0.1, 0.45)
  
  plt.show()