import pandas as pd
import torch
from transformers import RobertaTokenizer, T5ForConditionalGeneration, T5Tokenizer
import random
import pickle

# Step 1: Load the dataset
# Adjust the file path if necessary
data_path = '../../src/sample_dataset.csv'
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


# Step 3: Define a function to mask 15% of tokens in each input string, and return a list of masked words
def mask_tokens(input_text, mask_num = 15):
    # input_ids = T5CODE_TOKENIZER(input_text, return_tensors="pt").input_ids
    # print("input_ids,", input_ids)
    # print("=" * 200)

    # tokenized_strings = T5CODE_TOKENIZER.convert_ids_to_tokens(input_ids[0])
    # print("Tokenized strings:", tokenized_strings)
    # print("=" * 200)

    # decoded_text = T5CODE_TOKENIZER.decode(input_ids[0], skip_special_tokens=True)
    # print("Decoded text:", decoded_text)
    # print("=" * 200)

    input_text_list=input_text.split()

    text_length = len(input_text_list)
    ground_truth_ids = random.sample( range(text_length), mask_num )

    ground_truth_tokens= [input_text_list[i] for i in ground_truth_ids]

    for i in ground_truth_ids:
        input_text_list[i] = T5CODE_TOKENIZER.mask_token
    masked_text = " ".join(input_text_list)

    return masked_text, ground_truth_tokens

# Apply masking to each input
masked_text_list = []
ground_truth_tokens_list = []
for text in input_texts:
    one_masked_text, one_ground_truth_tokens = mask_tokens(text)
    masked_text_list.append(one_masked_text)
    ground_truth_tokens_list.append(one_ground_truth_tokens)

print("=" * 200)
for idx, masked_text in enumerate(masked_text_list):
    print(f"[{idx}] asked text: {masked_text}")
print("=" * 200)

for idx, ground_truth_token in enumerate(ground_truth_tokens_list):
    print(f"[{idx}] ground_truth_tokens: {ground_truth_token}")

print("masked_texts_length",len(masked_text_list))
print("=" * 200)

#save to pickle:
data_to_save = {
    "masked_text_list": masked_text_list,
    "ground_truth_tokens_list": ground_truth_tokens_list
}

# Save as a pickle file
with open("30_sample_data.pkl", "wb") as file:
    pickle.dump(data_to_save, file)

print("Data saved as '30_sample_data.pkl'")

# read the pickle file

with open("30_sample_data.pkl", "rb") as file:
    loaded_data = pickle.load(file)

masked_text_list = loaded_data["masked_text_list"]
ground_truth_tokens_list = loaded_data["ground_truth_tokens_list"]

for idx, masked_text in enumerate(masked_text_list):
    print(f"[{idx}] asked text: {masked_text}")
print("=" * 200)

for idx, ground_truth_token in enumerate(ground_truth_tokens_list):
    print(f"[{idx}] ground_truth_tokens: {ground_truth_token}")

print("masked_texts_length",len(masked_text_list))
print("=" * 200)

print("Data loaded successfully.")
