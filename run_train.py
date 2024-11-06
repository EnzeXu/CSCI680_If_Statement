import os.path
import pickle
import torch
import pandas as pd
import torch.nn as nn
from transformers import RobertaTokenizer, T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader, random_split


from ifstatement.model import *


def save_dataset_to_csv(one_dataset, save_path):
    target_block_list = []
    input_method_list = []
    data_loader = DataLoader(one_dataset, batch_size=1, shuffle=True)
    for batch in data_loader:
        _, _, x, y = batch
        x = x[0]
        y = [item[0] for item in y]
        # print(f"x: {x}")
        # print(f"y: {y}")
        input_method_list.append(x)
        target_block_list.append(" ".join(y))
    df = pd.DataFrame(columns=["id", "input_method", "target_block"])
    df["input_method"] = input_method_list
    df["target_block"] = target_block_list
    df['id'] = range(len(df))
    df.to_csv(save_path, index=False)


def one_time_run_pretrain():
    gpu_id = 2

    # Check if CUDA (GPU) is available and set the device accordingly
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Using GPU: {torch.cuda.get_device_name(gpu_id)} (ID: {gpu_id})")
        print(f"Memory available: {torch.cuda.get_device_properties(gpu_id).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    with open("src/full_dataset_train.pkl", "rb") as file:
        loaded_data = pickle.load(file)

    test_cut = 1000

    masked_text_list = loaded_data["X"]#[test_cut:2*test_cut]
    ground_truth_tokens_list = loaded_data["Y"]#[test_cut:2*test_cut]

    # print(f"Example masked_text_list[0]: {masked_text_list[0]}")
    # print(f"Example ground_truth_tokens_list[0]: {ground_truth_tokens_list[0]}")

    # Create dataset and DataLoader
    dataset = TextDataset(masked_text_list, ground_truth_tokens_list)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    print(f"Data Size [train/val/test]: {train_size}/{val_size}/{test_size}")

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    save_dataset_to_csv(test_dataset, "result/test_dataset.csv")


    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # for one_X, one_Y in train_loader:
    #     print(f"one_X: {one_X}")
    #     print(f"one_Y: {one_Y}")

    # Initialize the model and run training
    predictor = MaskPredictorModel().to(device)
    # predictor.train_model(train_loader, epochs=100, lr=1e-2)
    #
    # #save model:
    # weights = {
    #     "state_dict": predictor.model.state_dict()
    # }
    # if not os.path.exists("save"):
    #     os.makedirs("save")
    # torch.save(weights, os.path.join("save", "train_last.pth"))


if __name__ == "__main__":
    one_time_run_pretrain()
