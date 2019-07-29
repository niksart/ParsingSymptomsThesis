#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sklearn.metrics.pairwise as sk
import numpy as np

# --- PATHS ---
BASEDIR = "/home/niksart/ParsingSymptomsThesis/"

JSON_SYMPTOM_TREE_PATH = BASEDIR + "data/json/symptom_tree.json"

CSV_PATH = BASEDIR + "data/csv/"
CSV_GLOVE_PATH = CSV_PATH + "glove/"
CSV_WORD_LIST_GLOVE_40000 = CSV_GLOVE_PATH + "40000_word_list.csv"
CSV_ST_CONCEPT_NAMES_GLOVE_PATH = CSV_GLOVE_PATH + "symptom_tree/concept_names/"
CSV_ST_WORDS_GLOVE_PATH = CSV_GLOVE_PATH + "symptom_tree/words/"
CSV_BODY_PARTS_FILE = BASEDIR + "data/csv/hot_words/body_parts.csv"
CSV_ST_MEDICAL_WORDS_GLOVE_PATH = CSV_GLOVE_PATH + "symptom_tree/medical_words/"
CSV_ST_MEDICAL_TERMS_FILE = CSV_PATH + "symptom_tree/medical_terms.csv"

TXT_PATH = BASEDIR + "data/txt/"
TXT_GLOVE_PATH = TXT_PATH + "glove/"

JSON_FILE_QUESTIONS_QA = BASEDIR + "data/json/qa_questions.json"

DATASET_CV_FOLDER = BASEDIR + "dataset/cross_validation/"
DATASET_TRAINING_FOLDER = BASEDIR + "dataset/testing/"

STANFORD_NLP_RESOURCES_PATH = BASEDIR + "data/stanfordnlp_resources/"

ADMITTED_REPRESENTATIONS = ["glove"]

SIMILARITY_MEASURE_SUPPORTED = ["cosine", "euclidean"]
# Index of the aforementioned list
SIMILARITY_MEASURE = 0



# --- GLOVE SETTINGS ---
DEFAULT_NUM_IMPORTED_GLOVE_VECTORS = 40000
MAX_NUM_IMPORTED_GLOVE_VECTORS = 100000
POSSIBLE_GLOVE_DIMENSIONS = [50, 100, 200, 300]
DEFAULT_GLOVE_DIMENSION = 1 # index of the possible dimensions list


def strip_string(s):
    ret = ""
    for c in s:
      if not str(c).isalpha():
        ret += " "
      else:
        ret += c
    
    return ret


"""
Similarity between two matrices

A is #examplesA x #features
B is #examplesB x #features

"""
def similarity_matrices(A, B):
  if SIMILARITY_MEASURE_SUPPORTED[SIMILARITY_MEASURE] == "cosine":
    return sk.cosine_similarity(A, B)
  elif SIMILARITY_MEASURE_SUPPORTED[SIMILARITY_MEASURE] == "euclidean":
    euclidean_distances = sk.euclidean_distances(A, B)
    return np.power(np.e, - euclidean_distances)
