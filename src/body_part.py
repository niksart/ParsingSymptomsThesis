import commons as COM
import pandas as pd


"""
  Representation of a body part.
  
  id_bp: univoque id
  names: list of names of that bp
  ref_to: this bp is part of another bigger bp (pointed by ref_to)
  is_external: True if the bp is external
  not_specifically_located: True if the bp is not specifically located in a part of the body (e.g. for blood)
  one_word: True if all the names of the bp are made of one word
"""


class BodyPart:
  

  def __init__(self, *args):
    args_list = list(args)
    if len(args_list) == 6:
      self.id_bp = args_list[0]
      self.names = self.__get_list_of_names(args_list[1])
      self.ref_to = args_list[2]
      self.is_external = False if args_list[3] == 0 else True
      self.not_specifically_located = False if args_list[4] == 0 else True
      self.one_word = False if args[5] == 0 else True
    elif len(args_list) == 1:
      id_bp = args_list[0]
      self.__initialize_fields_given_id(id_bp)
    else:
      raise Exception("The number of parameters should be 1 (get a body part by id) or 6. \
                      See the documentation for further information.")
    
  
  def __get_list_of_names(self, names_text):
    return names_text.split("/")
  
  
  def __initialize_fields_given_id(self, id_bp):
    #carica file e lista di body parts
    self.body_parts = []
    body_parts_df = pd.read_csv(COM.CSV_BODY_PARTS_FILE)
    for _, row in body_parts_df.iterrows():
      if row["ID"] == id_bp:
        self.id_bp = id_bp
        self.names = self.__get_list_of_names(row["NAME"])
        self.ref_to = row["REF_TO"]
        self.is_external = False if row["IS_EXTERNAL"] == 0 else True
        self.not_specifically_located = False if row["NOT_SPECIFICALLY_LOCATED"] == 0 else True
        self.one_word = False if row["ONE_WORD"] == 0 else True
