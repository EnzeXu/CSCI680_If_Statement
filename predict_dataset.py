import os.path
from transformers import BertTokenizer, BertModel
import pandas as pd
import torch
import re
import numpy as np
from tqdm import tqdm
from transformers import RobertaTokenizer, T5Tokenizer, T5ForConditionalGeneration

from ifstatement.metric import get_cosine_similarity, sequence_matcher


def extract_if_pattern(s):
    # Find a substring that starts with "if", followed by any characters, and ends with ":"
    s = s.lstrip()
    if s[:4] == "elif" and s[4] != " ":
        s = "elif " + s[4:]
    if s[:2] == "if" and s[2] != " ":
        s = "if " + s[2:]
    match = re.search(r'elif.*?:', s)
    if match:
        return match.group()
    match = re.search(r'if.*?:', s)
    if match:
        return match.group()
    return s  # Return None if no match is found

def unit_predict(input_text, tokenizer, model):
    with torch.no_grad():
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        outputs = model.generate(input_ids, max_length=32, num_beams=5, early_stopping=True)
    predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predicted_text = extract_if_pattern(predicted_text)
    return predicted_text


def predict_dataset(input_path, model_load_path=None):
    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
    model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    if model_load_path:
        # assert os.path.exists(model_load_path)
        try:
            state_dict = torch.load(model_load_path)
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"[Error] {model_load_path} not found:", e)
    df = pd.read_csv(input_path)
    input_list = []
    correct_list = []
    expected_if_condition_list = []
    predicted_if_condition_list = []
    prediction_score_1_embedding_cosine_similarity_list = []
    prediction_score_2_sequence_matcher_list = []
    prediction_score_avg = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        input_text = row["input_method"]
        target_block = row["target_block"]
        target_block_hat = unit_predict(input_text, tokenizer, model)
        score_1 = round((get_cosine_similarity(target_block_hat, target_block, bert_tokenizer, bert_model) + 1) / 2.0 * 100, 2)
        score_2 = round(sequence_matcher(target_block_hat, target_block) * 100, 2)
        # print(f"target_block: {target_block}")
        # print(f"target_block_hat: {target_block_hat}")
        # print(f"cos_similarity_score: {score_1}")
        # print(f"sequence_matcher_score: {score_2}")
        # print("=" * 200)
        input_list.append(input_text)
        correct_list.append(str(score_1 > 99.99 or score_2 > 99.99).upper())
        expected_if_condition_list.append(target_block)
        predicted_if_condition_list.append(target_block_hat)
        prediction_score_1_embedding_cosine_similarity_list.append(score_1)
        prediction_score_2_sequence_matcher_list.append(score_2)
        prediction_score_avg.append(round((score_1 + score_2) / 2, 2))
    print(f"input_path: {input_path}")
    print(f"count: {len(input_list)}")
    print(f"correct: {correct_list.count('TRUE')} / {len(input_list)} = {100.0 * correct_list.count('TRUE') / len(input_list):.2f}%")
    print(f"score_1_embedding_cosine_similarity: avg={np.mean(np.array(prediction_score_1_embedding_cosine_similarity_list)):.2f}, std={np.std(np.array(prediction_score_1_embedding_cosine_similarity_list)):.2f}")
    print(f"score_2_sequence_matcher_list: avg={np.mean(np.array(prediction_score_2_sequence_matcher_list)):.2f}, std={np.std(np.array(prediction_score_2_sequence_matcher_list)):.2f}")
    print(f"score_avg: avg={np.mean(np.array(prediction_score_avg)):.2f}, std={np.std(np.array(prediction_score_avg)):.2f}")

    df_result = pd.DataFrame({
        "id": range(len(input_list)),
        "input": input_list,
        "correct": correct_list,
        "expected_if_condition": expected_if_condition_list,
        "predicted_if_condition_list": predicted_if_condition_list,
        "prediction_score_1(embedding_cosine_similarity)": prediction_score_1_embedding_cosine_similarity_list,
        "prediction_score_2(sequence_matcher)": prediction_score_2_sequence_matcher_list,
        "prediction_score(avg)": prediction_score_avg,
    })
    save_path = input_path.replace(".csv", "_prediction.csv")
    df_result.to_csv(save_path, index=False)


if __name__ == "__main__":
    predict_dataset("result/test_dataset.csv", "save/last.pth")
    # predict_dataset("result/sample.csv", "save/last.pth")
