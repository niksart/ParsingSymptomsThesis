import commons as COM
import anytree as at
import os
import pandas as pd
import numpy as np
from anytree.importer import DictImporter
from anytree.importer import JsonImporter

fh = open(COM.JSON_SYMPTOM_TREE_PATH, "r")
symptom_json = fh.read()


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
  Pass to it a Vectorifier object
  """
  def __init__(self,
               vectorifier=None):
    
    dict_importer = DictImporter(nodecls=Node)
    importer = JsonImporter(dictimporter=dict_importer)
    self.root = importer.import_(symptom_json)
    
    if vectorifier is not None:
      # normal mode, if it is None is eval mode
      self.vectorifier = vectorifier
      self.vector_dimension = vectorifier.d
      
      """
      if concept name vectors file not exists, create it
      """
      if vectorifier.internal_representation == "glove":
        if not os.path.isfile(COM.CSV_ST_CONCEPT_NAMES_GLOVE_PATH + \
                              str(self.vector_dimension) + "d.csv"):
          print("Concept names glove vectors file not found. \nComputing file...")
          self.__save_csv_concept_name_glove_vectors()
      
        self.concept_name_vectors_df = pd.read_csv(COM.CSV_ST_CONCEPT_NAMES_GLOVE_PATH + \
                                                   str(self.vector_dimension) + "d.csv", 
                                                   header=None)
      elif vectorifier.internal_representation == "bert":
        if not os.path.isfile(COM.CSV_ST_CONCEPT_NAMES_BERT_PATH + \
                              COM.FILENAME_CSV_ST_CONCEPT_NAMES_BERT):
          print("Concept names bert vectors file not found. \nComputing file...")
          self.__save_csv_concept_name_bert_vectors()
      
        self.concept_name_vectors_df = pd.read_csv(COM.CSV_ST_CONCEPT_NAMES_BERT_PATH + \
                                                   COM.FILENAME_CSV_ST_CONCEPT_NAMES_BERT, 
                                                   header=None)
    
    

  """
  GloVe vectors of concept names are precomputed and saved in a csv file.
  If the file not exists, compute them and save them. 
  """
  def __save_csv_concept_name_glove_vectors(self):
    df = self.__get_df_concept_names_vectors(self.root)

    os.makedirs(COM.CSV_ST_CONCEPT_NAMES_GLOVE_PATH, exist_ok=True)
    
    df.to_csv(COM.CSV_ST_CONCEPT_NAMES_GLOVE_PATH + \
              str(self.vector_dimension) + "d.csv", 
              index=False, header=False)
  
  
  """
  Bert vectors of concept names are precomputed and saved in a csv file.
  If the file not exists, compute them and save them. 
  """
  def __save_csv_concept_name_bert_vectors(self):
    df = self.__get_df_concept_names_vectors(self.root)

    os.makedirs(COM.CSV_ST_CONCEPT_NAMES_BERT_PATH, exist_ok=True)
    
    df.to_csv(COM.CSV_ST_CONCEPT_NAMES_BERT_PATH + COM.FILENAME_CSV_ST_CONCEPT_NAMES_BERT, 
              index=False, header=False)
  
  
  def __get_df_concept_names_vectors(self, parent):
    parent_vector = self.vectorifier.vectorize_concept_name(parent.get_concept_name())
    
    if not parent_vector.empty:
      df = pd.DataFrame(parent_vector).T
      df.insert(0, "concept_name", parent.get_concept_name())
      df.insert(1, "cui", parent.name)
    else:
      df = pd.DataFrame()
    
    for child in parent.children:
      df = df.append(self.__get_df_concept_names_vectors(child))
      
    return df
  
  
    
  def get_most_similar_cui_given_body_part_token(self,
                                                 token,
                                                 body_part,
                                                 min_similarity=0):
    
    return self.__get_most_similar_cui_given_token(token,
                                                   min_similarity,
                                                   body_part)


  def get_most_similar_cui_given_symptom_token(self,
                                               token,
                                               min_similarity=0):
    
    return self.__get_most_similar_cui_given_token(token, min_similarity)

  
  """
  Given a token from the Answer Interpreter, calculate the subsentences within it
  (all the possible subsets of the token) and get a list of tuples like this:
  
    ('cui', 0.89)
    number cui, similarity
    
  If min_similarity is passed, a tuple is produced only if the best match has
  the similarity > min_similarity. Otherwise, it returns None.
  
  If body_part is passed then search only in the symptoms related to that body part
  """
  def __get_most_similar_cui_given_token(self,
                                         token,
                                         min_similarity=0,
                                         body_part=None):
    subsentences = self.vectorifier.get_subsentences_and_vectors(token)
    
    # there are no vectorizable subsentences
    if subsentences == []:
      return {"found": "no"}
    
    if body_part is None:
      concept_names_vectors = self.concept_name_vectors_df
    else:
      concept_names_vectors = self.__get_concept_names_vectors_related_to_body_part(body_part)
    
    symptoms_df = self.__get_most_similar_symptoms_given_subsentences(subsentences,
                                                                      concept_names_vectors)
    
    symptoms_df = symptoms_df.sort_values("similarity", ascending=False)
    
    _, cui = symptoms_df.iloc[0][1]
    simil = symptoms_df.iloc[0][2]
    
    if simil > min_similarity:
      return {"found": "yes", "predicted_cui": cui, "similarity": simil}
    else:
      return {"found": "no"}
  
  
  """
  Function that returns the most suitable concept name for each subsentence,
  where 'suitable' here means more similar.
  
  Parameters in:
    - subsentences: list of ('subsentence', vector of the subsentence)
    - concept_names: dataframe representing concept names and their vector representations
  
  Returns:
    a pandas DataFrame with columns:
      - subsentence
      - tuple_symptom : ('concept name', CUI)
      - similarity
      => the number of rows is equal to the number of subsentences
  """
  def __get_most_similar_symptoms_given_subsentences(self,
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
    B = np.array(concept_names.iloc[:, 2:(self.vector_dimension+2)])
    dictB = {}
    list_concept_names = concept_names[0].tolist()
    list_cuis = concept_names[1].tolist()
    
    indexes = list(zip(list_concept_names, list_cuis))
    
    i = 0
    for concept_name, cui in indexes:
      dictB[i] = (concept_name, cui)
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
    
    return pd.DataFrame(ret, columns=["subsentence", "tuple_symptom", "similarity"])
  
  
  def __get_concept_names_vectors_related_to_body_part(self, body_part):
    list_root_nodes = body_part.root_nodes
    root_nodes = list(map(self.__get_node_by_concept_name, list_root_nodes))
    
    if root_nodes == []:
      # search in the whole tree
      root_nodes = [self.__get_node_by_concept_name("symptoms")]
    
    list_filtered_concept_names = []
    for root_node in root_nodes:
      l = self.__get_concept_names_of_subtree(root_node)
      list_filtered_concept_names += l
    
    selector = self.concept_name_vectors_df[0].isin(list_filtered_concept_names)
    return self.concept_name_vectors_df[selector]
  
      
  """
  
  ------------ UTILITIES ------------
  
  """
  
  def __get_node_by_concept_name(self, concept_name):
    return at.find(self.root, filter_=lambda node: node.get_concept_name()==concept_name)
  
  
  def __get_first_node_by_cui(self, cui):
    nodes = at.findall(self.root, filter_=lambda node: node.name==cui)
    if len(nodes) > 0:
      return nodes[0]
  
  
  def __get_concept_names_of_subtree(self, node):
    concept_names = [node.concept_name]
    
    for child in node.children:
      concept_names += self.__get_concept_names_of_subtree(child)
    
    return concept_names
  
  
  def __get_cuis_of_subtree(self, node):
    if node is not None:
      cuis = [node.name]
      
      for child in node.children:
        cuis += self.__get_cuis_of_subtree(child)
      
      return cuis
    else:
      return ["CUIplaceholder"]
  
  
  def get_cuis_subtree_given_cui_root(self, cui_root):
    root = self.__get_first_node_by_cui(cui_root)
    
    cuis = self.__get_cuis_of_subtree(root)
    
    return cuis
  
  
  def render_tree(self, root):
      return str(at.RenderTree(root, style=at.ContStyle()))
    
