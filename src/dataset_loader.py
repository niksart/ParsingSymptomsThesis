import commons as COM
import xml.etree.ElementTree as ET
import os


class DatasetLoader:
  
  
  def __init__(self):
    # find list paths CV
    # (path, filename)
    paths_cv = list(map(lambda s: (COM.DATASET_CV_FOLDER + s, os.path.splitext(s)[0]), 
                        os.listdir(COM.DATASET_CV_FOLDER)))
    # find list oaths TRAINING
    # (path, filename)
    paths_testing = list(map(lambda s: (COM.DATASET_TRAINING_FOLDER + s, os.path.splitext(s)[0]),
                             os.listdir(COM.DATASET_TRAINING_FOLDER)))
    
    self.sentences_cv = []
    for (path, filename) in paths_cv:
      with open(path) as fp:  
        line = fp.readline()
        while line:
          # interpreta XML
          root = ET.fromstring(line)
          sentence = list(self.__get_sentence_given_xml_root(root))
          self.sentences_cv.append((sentence, filename))
          # ---
          line = fp.readline()
    
    self.sentences_testing = []
    for (path, filename) in paths_testing:
      with open(path) as fp:  
        line = fp.readline()
        while line:
          # interpreta XML
          root = ET.fromstring(line)
          sentence = list(self.__get_sentence_given_xml_root(root))
          self.sentences_testing.append((sentence, filename))
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
         
  
  def get_text_given_sentence(self, sentence):
    ret = ""
    
    for fragment in sentence:
      if fragment[0] == "text":
        ret += fragment[1]
      elif fragment[0] == "symptom":
        ret += fragment[2]
    
    return ret