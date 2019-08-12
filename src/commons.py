#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sklearn.metrics.pairwise as sk
import numpy as np

# --- PATHS ---
BASEDIR = "/home/niksart/ParsingSymptomsThesis/"

CSV_PATH = BASEDIR + "data/csv/"
CSV_GLOVE_PATH = CSV_PATH + "glove/"
CSV_BERT_PATH = CSV_PATH + "bert/"
CSV_WORD_LIST_GLOVE_100000 = CSV_GLOVE_PATH + "100000_word_list.csv"

CSV_ST_CONCEPT_NAMES_GLOVE_PATH = CSV_GLOVE_PATH + "symptom_tree/concept_names/"

DIMENSION_BERT_EMBEDDINGS = 768
CSV_ST_CONCEPT_NAMES_BERT_PATH = CSV_BERT_PATH
FILENAME_CSV_ST_CONCEPT_NAMES_BERT = "ST_concept_names.csv"
IP_BERT_SERVER = "51.15.134.168"

CSV_ST_BODY_PARTS_TAGGING_PATH = CSV_PATH + "symptom_tree/body_parts_tagging.csv"
CSV_BODY_PARTS_FILE = BASEDIR + "data/csv/hot_words/body_parts.csv"

TXT_PATH = BASEDIR + "data/txt/"
TXT_GLOVE_PATH = TXT_PATH + "glove/"

JSON_FILE_QUESTIONS_QA = BASEDIR + "data/json/qa_questions.json"
JSON_SYMPTOM_TREE_PATH = BASEDIR + "data/json/symptom_tree.json"
JSON_FILE_QASYSTEM_RNET_EVAL_TO_TAG = BASEDIR + "data/json/results_qa_evaluation/sentences_and_tokens_for_pred_RNET_to_tag.json"
JSON_FILE_QASYSTEM_BERT_EVAL_TO_TAG = BASEDIR + "data/json/results_qa_evaluation/sentences_and_tokens_for_pred_BERT_to_tag.json"
JSON_FILE_QASYSTEM_RNET_EVAL = BASEDIR + "data/json/results_qa_evaluation/sentences_and_tokens_for_pred_RNET.json"
JSON_FILE_QASYSTEM_BERT_EVAL = BASEDIR + "data/json/results_qa_evaluation/sentences_and_tokens_for_pred_BERT.json"
JSON_RESULTS_FOLDER = BASEDIR + "data/json/results/"

DATASET_CV_FOLDER = BASEDIR + "dataset/cross_validation/"
DATASET_TRAINING_FOLDER = BASEDIR + "dataset/testing/"

STANFORD_NLP_RESOURCES_PATH = BASEDIR + "data/stanfordnlp_resources/"

ADMITTED_REPRESENTATIONS = ["glove", "bert"]
QA_SYSTEM_TYPES = ["bert", "rnet"]

SIMILARITY_MEASURE_SUPPORTED = ["cosine", "euclidean"]
# Index of the aforementioned list
SIMILARITY_MEASURE = 0


# --- GLOVE SETTINGS ---
DEFAULT_NUM_IMPORTED_GLOVE_VECTORS = 100000
MAX_NUM_IMPORTED_GLOVE_VECTORS = 300000
POSSIBLE_GLOVE_DIMENSIONS = [50, 100, 200, 300]
DEFAULT_GLOVE_DIMENSION = 1 # index of the possible dimensions list


def strip_string(s):
    ret = ""
    for c in s:
      if not str(c).isalpha():
        ret += " "
      else:
        ret += c
    
    return ret.lower()


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
