# imports
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel

class BertEmbedder:
  
  
  def __init__(self):
    # Load pre-trained model tokenizer (vocabulary)
    self.__tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Load pre-trained model (weights)
    self.__model = BertModel.from_pretrained('bert-base-uncased')

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    self.__model.eval()
  
  
  """
  Given a sentence, calculate contextual embeddings.
  
  Two methods supported:
    - concat_last_4 (3072 dim) : embeddings are computed with a concatenation 
      of the last 4 hidden layers
      
    - sum_last_4 (768 dim) DEFAULT : embeddings are computed summing up the 
      last 4 hidden layers
  """
  def get_embeddings(self, sentence, type_embeddings = "sum_last_4"):
    marked_sentence = "[CLS] " + sentence + " [SEP]"
    # Tokenization
    tokenized_text = self.__tokenizer.tokenize(marked_sentence)
    # Indexing on the vocabulary ids
    indexed_tokens = self.__tokenizer.convert_tokens_to_ids(tokenized_text)
    # Words in the same sentence have the same segment_id
    segments_ids = [1] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Predict hidden states features for each layer
    # torch.no_grad because it is only a feed forward neural network
    with torch.no_grad():
      encoded_layers, _ = self.__model(tokens_tensor, segments_tensors)
    
    # Convert the hidden state embeddings into single token vectors
    # Holds the list of 12 layer embeddings for each token
    # Will have the shape: [# tokens, # layers, # features]
    token_embeddings = [] 
    
    # We have only one sentence
    batch_i = 0
    # For each token in the sentence...
    for token_i in range(len(tokenized_text)):
      
      # Holds 12 layers of hidden states for each token 
      hidden_layers = [] 
      
      # For each of the 12 layers...
      for layer_i in range(len(encoded_layers)):
        
        # Lookup the vector for `token_i` in `layer_i`
        vec = encoded_layers[layer_i][batch_i][token_i]
        
        hidden_layers.append(vec)
        
      token_embeddings.append(hidden_layers)
    
    # COMPUTE WORD EMBEDDINGS
    if type_embeddings == "concat_last_4":
      # 1- CONCATENATION OF LAST 4 LAYERS
      # Stores the token vectors, with shape [22 x 3,072]
      token_vecs_cat = []
      
      for token in token_embeddings:
          cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), 0)
        
          token_vecs_cat.append(cat_vec)
      
      return (tokenized_text, token_vecs_cat)
    
    elif type_embeddings == "sum_last_4":
      # 2 - SUMMATION OF LAST 4 LAYERS
      # Stores the token vectors, with shape [22 x 768]
      token_vecs_sum = []
      
      for token in token_embeddings:
          sum_vec = torch.sum(torch.stack(token)[-4:], 0)
          
          token_vecs_sum.append(sum_vec)
      
      return (tokenized_text, token_vecs_sum)
    else:
      raise Exception("type_embedding must be a supported type of embedding")
    