import os.path

import torch
import torch.nn as nn
from transformers import RobertaTokenizer, T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader, random_split
import pickle

from .model import TextDataset, MaskPredictorModel


# read the pickle file

# with open("30_sample_data.pkl", "rb") as file:
#     loaded_data = pickle.load(file)
#
# masked_text_list = loaded_data["masked_text_list"]
# ground_truth_tokens_list = loaded_data["ground_truth_tokens_list"]
#
# for idx, masked_text in enumerate(masked_text_list):
#     print(f"[{idx}] asked text: {masked_text}")
# print("=" * 200)
#
# for idx, ground_truth_token in enumerate(ground_truth_tokens_list):
#     print(f"[{idx}] ground_truth_tokens: {ground_truth_token}")
#
# print("masked_texts_length",len(masked_text_list))
# print("=" * 200)
#
# print("Data loaded successfully.")

# Sample input and target lists
masked_text_list = [
    "This is a sample with a <mask> and another <mask>.",
    "Another <mask> sentence with multiple <mask> tokens.",
    "Example with a <mask> in a different <mask> location.",
    "A sentence with <mask> placed <mask> differently.",
    "Here is one more <mask> example with <mask> tokens.",
    "Testing with <mask> and another <mask> in the text.",
    "The <mask> example includes multiple <mask> usages.",
    "This <mask> sample has several <mask> tokens inside.",
    "An input text <mask> with various <mask> places.",
    "Final example of <mask> text with masked <mask> tokens.",
]

ground_truth_tokens_list = [
    ["a", "b", "c"],
    ["c", "d"],
    ["e", "f"],
    ["g", "h"],
    ["token", "tokens"],
    ["token", "token"],
    ["token", "token"],
    ["token", "tokens"],
    ["token", "token"],
    ["token", "token"]
]

# Create dataset and DataLoader
dataset = TextDataset(masked_text_list, ground_truth_tokens_list)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])


batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for one_X, one_Y in train_loader:
    print(f"one_X: {one_X}")
    print(f"one_Y: {one_Y}")

# Initialize the model and run training
predictor = MaskPredictorModel()
predictor.train(train_loader, epochs=4, lr=5e-5)

#save model:
weights={"state_dict":predictor.model.state_dict()}
if not os.path.exists("save"):
    os.makedirs("save")
torch.save(weights, os.path.join("save", "30_sample_best.pth"))
#
# # Load the state_dict into a new model instance
# loaded_model = SimpleModel()
# loaded_model.load_state_dict(torch.load("simple_model_state_dict.pth"))
#
# # Verify that parameters are identical
# for param in loaded_model.state_dict():
#     print(param, "\t", torch.equal(loaded_model.state_dict()[param], model.state_dict()[param]))
