from transformers import BertTokenizer, BertModel
import torch
from torch.nn.functional import cosine_similarity

# Load pre-trained BERT model and tokenizer
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# model = BertModel.from_pretrained("bert-base-uncased")

def get_cosine_similarity(string_predict, string_gt, tokenizer, model):
    # Tokenize and encode input strings
    inputs_predict = tokenizer(string_predict, return_tensors="pt")
    inputs_gt = tokenizer(string_gt, return_tensors="pt")

    # Get the embeddings from the last hidden layer of BERT
    with torch.no_grad():
        embedding_predict = model(**inputs_predict).last_hidden_state.mean(dim=1)
        embedding_gt = model(**inputs_gt).last_hidden_state.mean(dim=1)

    # Compute cosine similarity
    similarity = cosine_similarity(embedding_predict, embedding_gt).item()
    return similarity

# # Example usage
# string_predict = "if ( a = 0 )"
# string_gt = "if ( a = )"
# similarity = get_cosine_similarity(string_predict, string_gt)
# print("Cosine Similarity:", similarity)
