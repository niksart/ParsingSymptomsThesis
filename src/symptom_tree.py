import commons as COM
import anytree as at
import os
import pandas as pd
import numpy as np
from vectorifier import Vectorifier
from anytree.importer import DictImporter
from anytree.importer import JsonImporter

fh = open(COM.JSON_SYMPTOM_TREE_PATH, "r")
symptom_json = fh.read()

SPEC = "specialization"
BASE = "base_concept"


class Node(at.node.node.Node):
  # concept_name identifies univoquely the node
  def get_concept_name(self):
    return COM.strip_string(self.concept_name)
  
  def __eq__(self, other):
    return self.get_concept_name() == other.get_concept_name()
  
  
  def __hash__(self):
    return hash(self.__repr__())
  
  
  
class SymptomTree:
  
  """
  internal_representation: can be "glove" or "bert"
  
  GloVe parameters --
    number_glove_words: take the first n words from GloVe dictionary (they are ordered by frequency)
    dimension_glove_vectors: can be 50, 100, 200, 300
  """
  
  def __init__(self,
               internal_representation,
               number_glove_words = COM.DEFAULT_NUM_IMPORTED_GLOVE_VECTORS,
               dimension_glove_vectors = COM.POSSIBLE_GLOVE_DIMENSIONS[COM.DEFAULT_GLOVE_DIMENSION]):
    
    if internal_representation not in COM.ADMITTED_REPRESENTATIONS:
      raise Exception("A proper internal representation should be specified.\n"
                      "Supported representations: " + ", ".join(COM.ADMITTED_REPRESENTATIONS))
    
    dict_importer = DictImporter(nodecls=Node)
    importer = JsonImporter(dictimporter=dict_importer)
    self.root = importer.import_(symptom_json)
    self.__mark_specializations(self.root)
    self.__mark_base_concepts(self.root)
    
    self.dimension_glove_vectors = dimension_glove_vectors
    self.vectorifier = Vectorifier(internal_representation,
                                        number_glove_words,
                                        dimension_glove_vectors)
    
    # if concept name vectors file not exists, create it
    # this file contains the glove vectors of the concept names
    if not os.path.isfile(COM.CSV_ST_CONCEPT_NAMES_GLOVE_PATH + \
                          str(self.dimension_glove_vectors) + "d.csv"):
      self.__save_csv_concept_name_vectors()
    
    self.concept_name_vectors_df = pd.read_csv(COM.CSV_ST_CONCEPT_NAMES_GLOVE_PATH + \
                                               str(self.dimension_glove_vectors) + "d.csv", 
                                               header=None)
    

  """
  Vectors of concept names are precomputed and saved in a csv file.
  If the file not exists, compute them and save them. 
  """
  def __save_csv_concept_name_vectors(self):
    df = self.__get_df_concept_names_vectors(self.root)

    os.makedirs(COM.CSV_ST_CONCEPT_NAMES_GLOVE_PATH, exist_ok=True)
    
    df.to_csv(COM.CSV_ST_CONCEPT_NAMES_GLOVE_PATH + \
              str(self.dimension_glove_vectors) + "d.csv", 
              index=False, header=False)
  
  
  def __get_df_concept_names_vectors(self, parent):
    parent_vector = self.vectorifier.vectorize_concept_name(parent.get_concept_name())
    
    if not parent_vector.empty:
      df = pd.DataFrame(parent_vector).T
      df.insert(0, "concept_name", parent.get_concept_name())
    else:
      df = pd.DataFrame()
    
    for child in parent.children:
      df = df.append(self.__get_df_concept_names_vectors(child))
      
    return df
  
  
  """
  Given a sentence of the patient, calculate the subsentences and get a 
  list of tuples like this: 
  
    ('subsentence', 'most similar concept name', 0.89)
  
  """
  def get_most_similar_concept_name_given_sentence(self, sentence):
    subsentences = self.vectorifier.get_subsentences_and_vectors(sentence)
    
    return self.__get_most_similar_concept_name_given_subsentences(subsentences,
                                                                   self.concept_name_vectors_df)
  
  
  """
  Function that returns the most suitable concept name for each subsentence,
  where 'suitable' here means more similar.
  
  Parameters in:
    - subsentences: list of ('subsentence', vector of the subsentence)
    - concept_names: dataframe representing concept names and their vector representations
  
  Returns:
    a list of tuples ('subsentence', 'most similar concept name', 0.89)
      => the number of tuples is equal to the number of subsentences
  """
  def __get_most_similar_concept_name_given_subsentences(self,
                                                       subsentences,
                                                       concept_names):
    # Matrix that has the vectors of subsentences on rows
    # shape of A is #subsentences x #dimensions_inter_repr
    A = np.array([list(vect) for (_, vect) in subsentences])
    # lists preserves the visit order
    dictA = {}
    i = 0
    for (subsentence, _) in subsentences:
      dictA[i] = subsentence
      i += 1
    
    # Matrix that has the vectors of concept names on rows
    # shape of B is #concept_names x #dimensions_inter_repr
    B = np.array(concept_names.iloc[:, 1:(self.dimension_glove_vectors+1)])
    dictB = {}
    list_concept_names = concept_names[0].tolist()
    i = 0
    for concept_name in list_concept_names:
      dictB[i] = concept_name
      i += 1
      
    similarity_matrix = COM.similarity_matrices(A, B)
    
    indexes_max_rows = similarity_matrix.argmax(axis=1)
    
    ret = []
    i_subsentence = 0
    for i_concept_name in indexes_max_rows:
      ret.append((dictA[i_subsentence], 
                  dictB[i_concept_name], 
                  similarity_matrix[i_subsentence][i_concept_name]))
      i_subsentence += 1
    
    return ret
  
      
  """
  
  ------------ UTILITIES ------------
  
  """
  
  def get_node_by_concept_name(self, concept_name):
    return at.find(self.root, filter_=lambda node: node.get_concept_name()==concept_name)
  
  
  def render_tree(self, root):
      return str(at.RenderTree(root, style=at.ContStyle()))
    
  
  def count_nodes(self, parent, type_nodes=""):    
    counter = 0
    if type_nodes == "" or (hasattr(parent, "type") and parent.type == type_nodes):
      counter = 1
    
    for child in parent.children:
      counter += self.count_nodes(child, type_nodes)
    
    return counter
  
  
  # true if the two strings has at least a word in common
  def __have_words_in_common(self, s1, s2):
    # these words do not cause intersection
    words_of_no_intersection = set(["symptoms"])
    
    set_s1 = set(s1.lower().split())
    set_s2 = set(s2.lower().split())
    
    return (set_s1.intersection(set_s2) - words_of_no_intersection) != set()
  
  
  """
  
  ------------ TAGGING NODES ------------
  
  """
  
  """
  1 - mark specialization
  
  Given a tree rooted in `root`, if a node has a concept name that has at 
  least a word in common with an ancestor, and that ancestor does not contain
  inside the name the word `symptoms`:
    ADD an attribute named `type` equal to SPEC (that node is a specialization)
    ADD an attribute `specialization_of` that points to that ancestor 
  """
  def __mark_specializations(self, parent):
    # consider the ancestors starting from the father
    for ancestor in parent.ancestors[::-1]:
       # if a word in common AND ancestor does not contain the word `symptoms`
      if self.__have_words_in_common(parent.get_concept_name(), ancestor.get_concept_name()) \
      and (ancestor.get_concept_name().find("symptoms") ==  -1):
        parent.type = SPEC
        parent.specialization_of = ancestor
        break
    
    for child in parent.children:
      self.__mark_specializations(child)
  
  
  """
  2 - mark the base concepts. A base concept is behind the front of specializations
  
  DFS, when you find a specialization, stop the recursion over the children
  """
  def __mark_base_concepts(self, parent):
    if not hasattr(parent, "type"):
      parent.type = BASE
      for child in parent.children:
        self.__mark_base_concepts(child)
  
  
  """
  
  ------------ LCS within the specialization tree ------------
  
  To prove the meaningfulness of specializations:
  - find the longest common substring within the tree rooted in the first
    specialization
  - look at percentages of presence of this lcs
  
  (maybe lcs is not the best measure)
  """
  
  
  """
  Given a tree rooted in parent, returns a list with the concept names of the nodes
  
  The list returned is the input for the __lcs function
  """
  def __concept_names_by_root(self, parent):
    ret = [parent.get_concept_name()]
    
    for child in parent.children:
      ret = ret + self.__concept_names_by_root(child)
    
    return ret
  

  """
  Longest common substring
  data: list of strings
  
  https://stackoverflow.com/questions/2892931/longest-common-substring-from-more-than-two-strings-python
  """
  def __lcs(self, data):
    substr = ''
    if len(data) > 1 and len(data[0]) > 0:
      for i in range(len(data[0])):
        for j in range(len(data[0])-i+1):
          if j > len(substr) and all(data[0][i:i+j] in x for x in data):
            substr = data[0][i:i+j]
    return substr
  
  
  """
  Counts how many nodes of the tree rooted in parent have the string `s` in 
  their name_concept
  """
  def __count_string_in_name_concepts(self, s, parent):
    occurrence = not parent.get_concept_name().find(s) == -1
    
    if parent.is_leaf:
      if occurrence:
        return 1
      else:
        return 0
    
    for child in parent.children:
      if occurrence:
        return 1 + self.__count_string_in_name_concepts(s, child)
      else:
        return self.__count_string_in_name_concepts(s, child)
      
  
  """
  Find the lcs in the subtree pointed by `root` and calculate the percentage of
  occurrence on the nodes
  """
  def find_lcs_given_root(self, root):
    lcs = self.__lcs(self.__concept_names_by_root(root))    
    percentage = self.__count_string_in_name_concepts(lcs, root) / self.count_nodes(root) * 100
    
    return (root, lcs, str(self.__count_string_in_name_concepts(lcs, root)) + " / " + str(self.count_nodes(root)), percentage)
  
  """
  Find the set of nodes that are the `specialization_of` of the first 
  specialization nodes (the front of specializations).
  They are the last base concepts and are always ancestors of specializations
  """
  def find_roots_of_specializations(self, root):
    front_concept_names = self.__find_roots_of_specializations_rec(root)
    front_nodes_set = map(self.get_node_by_concept_name, front_concept_names)
    return tuple(front_nodes_set)
  
  
  def __find_roots_of_specializations_rec(self, parent):
    front_concept_names = set()
    
    if hasattr(parent, "type") and parent.type == SPEC:
      return { parent.specialization_of.get_concept_name() }
    
    for child in parent.children:
      front_concept_names = front_concept_names.union(self.__find_roots_of_specializations_rec(child))
    
    return front_concept_names
  