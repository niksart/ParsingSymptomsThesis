import anytree as at
import commons as com
import pandas as pd
from anytree.exporter import JsonExporter

symptom_df =pd.read_csv(com.BASEDIR + "csvs/symptom_tree_MEDCIN/symptom_tree_MEDCIN.csv")


def add_children(parent_node, parent_aui):
    print("add children of: ", parent_node)
    for index in (symptom_df.index[symptom_df['PAUI'] == parent_aui].tolist()):
        child_aui = symptom_df.iloc[index, 0] # AUI is the first column
        child_cui = symptom_df.iloc[index, 2] # where here CUI is the third column
        child_name = symptom_df.iloc[index, 3] # the name of the child
        child_node = at.Node(child_cui, parent=parent_node, concept_name=child_name)
    
        add_children(child_node, child_aui)


root_cui = "C1457887"
root_aui = "A21010092"
root_node = at.Node(root_cui, concept_name="symptoms")

add_children(root_node, root_aui)

exporter = JsonExporter(indent=2, sort_keys=True)
fh = open("/home/niksart/...", "w")
exporter.write(root_node, fh)
