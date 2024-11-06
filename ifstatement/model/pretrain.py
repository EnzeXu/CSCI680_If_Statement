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
input_texts = df['input_method_all'].tolist()
# for input_text in input_texts:
#     print("unprocessed_input_texts: ", input_text)
# print(len(input_texts))  # 30


# Step 2: Load the pre-trained tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5p-220m")  # Load CodeT5+ tokenizer
# model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5p-220m")
T5CODE_TOKENIZER = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
# T5CODE_MODEL = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')
KEYWORDS = ["if", "elif", "else", "for", "while", "try", "except", "with", "def", "return", "raise"]
random.seed(0)

def mask_periods(input_text_list, mask_ratio=0.15):
    mask_num = 30  # int(mask_ratio * len(input_text_list))
    len_text = len(input_text_list)
    masked_strings = []
    input_text_list_masked = [item for item in input_text_list]

    # List to store non-overlapping mask periods
    mask_periods = []
    z = 0
    z_limit = 1000
    while len(mask_periods) < mask_num and z < z_limit:
        # Randomly choose a start index, ensuring room for max period length and gap requirement
        start_idx = random.randint(0, len_text - 1)
        period_length = 1 # random.randint(4, 6)
        end_idx = start_idx + period_length

        # Ensure end_idx is within bounds
        if end_idx > len_text:
            end_idx = len_text

        # Check for overlap and gap with existing periods

        if all((start_idx != start) for start, end in mask_periods):
            mask_periods.append((start_idx, end_idx))
        z += 1

    mask_periods.sort(key=lambda x: x[0])
    for idx, (start, end) in enumerate(mask_periods):
        # Join the continuous strings for masked strings
        masked_strings.append(input_text_list[start])

        # Replace the period with "<mask>" in the new masked list
        assert input_text_list_masked[start] != "<mask>"
        # print(f"<mask{idx}> was {input_text_list_masked[start]}")
        input_text_list_masked[start] = f"<mask>"

    # print(f"mask_periods: {mask_periods}")
    # print("-" * 200)
    # print(f"masked_strings: {masked_strings}")
    # print("-" * 200)
    # print(f"input_text_list: {input_text_list}")
    # print("-" * 200)
    # print(f"input_text_list_masked: {input_text_list_masked}")
    # print("=" * 300)

    return masked_strings, input_text_list_masked

# Step 3: Define a function to mask 15% of tokens in each input string, and return a list of masked words
def mask_tokens(input_text):
    # input_ids = T5CODE_TOKENIZER(input_text, return_tensors="pt").input_ids
    # print("input_ids,", input_ids)
    # print("=" * 200)

    # tokenized_strings = T5CODE_TOKENIZER.convert_ids_to_tokens(input_ids[0])
    # print("Tokenized strings:", tokenized_strings)
    # print("=" * 200)

    # decoded_text = T5CODE_TOKENIZER.decode(input_ids[0], skip_special_tokens=True)
    # print("Decoded text:", decoded_text)
    # print("=" * 200)

    input_text = input_text.replace("<TAB> ", "").replace(" <TAB>", "") # Enze: I want to remove all <TAB> tokens
    input_text_list = input_text.split()

    # text_length = len(input_text_list)
    # ground_truth_ids = random.sample( range(text_length), mask_num )

    ground_truth_tokens, input_text_list_masked = mask_periods(input_text_list, mask_ratio=0.15)

    # ground_truth_tokens = [input_text_list[i] for i in ground_truth_ids]

    # for i in ground_truth_ids:
    #     input_text_list[i] = T5CODE_TOKENIZER.mask_token
    masked_text = " ".join(input_text_list_masked)

    return input_text, masked_text, ground_truth_tokens

# Apply masking to each input
masked_text_list = []
ground_truth_tokens_list = []
for text in tqdm(input_texts):
    one_raw_text, one_masked_text, one_ground_truth_tokens = mask_tokens(text)
    # print("=" * 200)
    # print("=" * 200)
    # print(f"one_raw_text: {one_raw_text}")
    # print("-" * 200)
    # print(f"one_masked_text: {one_masked_text}")
    # print("-" * 200)
    # print(f"one_ground_truth_tokens: {one_ground_truth_tokens}")
    if len(one_masked_text.split()) >= 100:
        masked_text_list.append(one_masked_text)
        ground_truth_tokens_list.append(one_ground_truth_tokens)

# print("=" * 200)
# for idx, masked_text in enumerate(masked_text_list):
#     print(f"[{idx}] asked text: {masked_text}")
# print("=" * 200)
#
# for idx, ground_truth_token in enumerate(ground_truth_tokens_list):
#     print(f"[{idx}] ground_truth_tokens: {ground_truth_token}")
#
# print("masked_texts_length",len(masked_text_list))
# print("=" * 200)

#save to pickle:
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
save_path = "src/full_dataset.pkl"
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
