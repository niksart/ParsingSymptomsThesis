import commons as COM
import xml.etree.ElementTree as ET
import os


class DatasetLoader:
  
  
  def __init__(self):
    # find list paths TESTING
    # (path, filename)
    paths_testing = list(map(lambda s: (COM.DATASET_TRAINING_FOLDER + s, os.path.splitext(s)[0]),
                             os.listdir(COM.DATASET_TRAINING_FOLDER)))
    
    self.__sentences_testing = []
    for (path, filename) in paths_testing:
      with open(path) as fp:  
        line = fp.readline()
        while line:
          # interpreta XML
          root = ET.fromstring(line)
          sentence = list(self.__get_sentence_given_xml_root(root))
          self.__sentences_testing.append((sentence, filename))
          # ---
          line = fp.readline()
    
    
  """
  Given an XML root, returns a foldable of tuples of two types:
    - ("text", "...text...")
    - ("symptom", "C*******", "...text...")
  """
  def __get_sentence_given_xml_root(self, root):
     if root.text is not None:
       yield ("text", root.text)
     for child in root:
       if child.tag == "symptom":
         yield ("symptom", child.attrib["CUI"], child.text)
       if child.tail is not None:
         yield ("text", child.tail)
         
  
  def __get_text_given_sentence(self, sentence):
    ret = ""
    
    for fragment in sentence:
      if fragment[0] == "text":
        ret += fragment[1] if fragment[1] != None else ""
      elif fragment[0] == "symptom":
        ret += fragment[2] if fragment[2] != None else ""
    
    return ret
  
  
  def __get_cuis_list_given_sentence(self, sentence):
    ret = []
    
    for fragment in sentence:
      if fragment[0] == "symptom":
        ret.append(fragment[1])
    
    return ret
  
  
  """
  Returns a list of tuples
    [("sentence1_text", ["cui1", "cui2"]), (...), ..., (...)]
  """
  def get_testing_sentences_and_cuis(self):
    ret = []
    
    for sentence, type_sentence in self.__sentences_testing:
      sentence_text = self.__get_text_given_sentence(sentence)
      list_cuis = self.__get_cuis_list_given_sentence(sentence)
      
      ret.append((sentence_text, list_cuis))
    
    return ret
  
