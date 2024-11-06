import pandas as pd
import torch
from transformers import RobertaTokenizer, T5ForConditionalGeneration, T5Tokenizer
import random
import numpy as np
import pickle
from tqdm import tqdm

# Step 1: Load the dataset
# Adjust the file path if necessary
# data_path = '../../src/sample_dataset.csv'
data_path = 'src/full_dataset.csv'
df = pd.read_csv(data_path)
input_method_list = df['input_method_if_statement'].tolist()
target_block_list = df['target_block'].tolist()

random.seed(0)


# Apply masking to each input
masked_text_list = []
ground_truth_tokens_list = []
for one_input_method, one_target_block in tqdm(zip(input_method_list, target_block_list)):
    one_input_method = one_input_method.replace("<TAB> ", "").replace(" <TAB>", "")
    one_ground_truth_tokens = one_target_block.split()
    FILL_IN = "<fill-in>"
    assert FILL_IN in one_input_method
    one_masked_text = one_input_method.replace(FILL_IN, " ".join(["<mask>"] * len(one_ground_truth_tokens)))
    # print("=" * 200)
    # print("=" * 200)
    # # print(f"one_raw_text: {one_raw_text}")
    # print("-" * 200)
    # print(f"one_masked_text: {one_masked_text}")
    # print("-" * 200)
    # print(f"one_ground_truth_tokens: {one_ground_truth_tokens}")
    if len(one_masked_text.split()) >= 100:
        masked_text_list.append(one_masked_text)
        ground_truth_tokens_list.append(one_ground_truth_tokens)

masked_text_list = masked_text_list[:50000]
ground_truth_tokens_list = ground_truth_tokens_list[:50000]

data_to_save = {
    "X": masked_text_list,
    "Y": ground_truth_tokens_list
}
length_list = []
for idx, item in enumerate(masked_text_list):
    item_split = item.split()
    length_list.append(len(item_split))
    if len(item_split) < 30:
        print(f"[idx={idx} len={len(item_split)}] {item}")
# length_list = [len(item.split()) for item in masked_text_list]
print(f"count: {len(length_list)}")
print(f"max: {max(length_list)}")
print(f"min: {min(length_list)}")
print(f"sum: {sum(length_list)}")
print(f"std: {np.std(np.asarray(length_list))}")
print(f"avg: {sum(length_list) / len(length_list)}")

# Save as a pickle file
save_path = "src/full_dataset_train.pkl"
with open(save_path, "wb") as file:
    pickle.dump(data_to_save, file)

print(f"Data saved as {save_path}")

# read the pickle file
#
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
