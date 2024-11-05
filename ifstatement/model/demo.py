import os.path

import torch
import torch.nn as nn
from transformers import RobertaTokenizer, T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader, random_split
import pickle


TOKENIZER = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
MODEL = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')
# MaskPredictor class as defined
class MaskPredictor:
    def __init__(self, model_name='Salesforce/codet5-base'):
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    # def prepare_inputs(self, input_text):
    #     # Replace <mask> with the tokenizer's mask token
    #     input_text = input_text.replace("<mask>", self.tokenizer.mask_token)
    #     inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    #     return inputs

    def forward(self, inputs):
        # inputs = self.prepare_inputs(input_text)
        outputs = self.model.generate(
            inputs,  # inputs['input_ids'],
            max_length=50,  # Adjust max_length based on expected output
            num_return_sequences=1
        )
        return outputs

    def train(self, data_loader, epochs=3, lr=5e-5):
        # Define the optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=lr)
        num_training_steps = epochs * len(data_loader)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                    num_training_steps=num_training_steps)

        self.model.train()

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            total_loss = 0

            for batch in data_loader:
                input_tensor, target_tensor = batch
                input_tensor = input_tensor.to(self.model.device)
                target_tensor = target_tensor.to(self.model.device)
                # print(f"[input_tensor] {input_tensor.shape}")
                # print(f"[target_tensor] {target_tensor.shape}")

                # Flatten target_texts into strings for training
                # target_texts_flat = [" ".join(target) for target in target_texts]

                # Tokenize inputs and targets with consistent padding and max_length
                # inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=50)
                # targets = self.tokenizer(target_texts_flat, return_tensors="pt", padding=True, truncation=True,
                #                          max_length=50).input_ids

                # Move tensors to the appropriate device if using GPU
                # inputs = {key: val.to(self.model.device) for key, val in inputs.items()}
                # targets = targets.to(self.model.device)

                # Create `decoder_input_ids` from target tokens for T5
                decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(target_tensor)

                # Forward pass with labels for training
                # print(f"targets: {targets.shape}, {targets}")
                # print(f"inputs['input_ids']: {inputs['input_ids'].shape}, {inputs['input_ids']}")
                # print(f"decoder_input_ids: {decoder_input_ids.shape}, {decoder_input_ids}")
                outputs = self.model(input_tensor, decoder_input_ids=decoder_input_ids, labels=target_tensor)
                loss = outputs.loss

                # Backpropagation
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                total_loss += loss.item()

            print(f"Loss: {total_loss / len(data_loader)}")


# Define custom dataset
class TextDataset(Dataset):
    def __init__(self, masked_text_list, ground_truth_tokens_list):
        self.tokenizer = TOKENIZER
        # self.model = MODEL
        input_text = [item.replace("<mask>", self.tokenizer.mask_token) for item in masked_text_list]
        self.X = torch.tensor(self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).input_ids).clone().detach()
        # self.X = masked_text_list
        print(f"self.X: {self.X.shape}: {self.X}")
        self.Y = torch.tensor([self.tokenizer.convert_tokens_to_ids(item) for item in ground_truth_tokens_list])
        # self.Y = ground_truth_tokens_list
        print(f"self.Y: {self.Y.shape}: {self.Y}")
        # assert len(self.X) == len(self.Y)

    # def prepare_inputs(self, input_text):
    #     # Replace <mask> with the tokenizer's mask token
    #     input_text = input_text.replace("<mask>", self.tokenizer.mask_token)
    #     inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    #     return inputs

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


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

# # Sample input and target lists
# masked_text_list = [
#     "This is a sample with a <mask> and another <mask>.",
#     "Another <mask> sentence with multiple <mask> tokens.",
#     "Example with a <mask> in a different <mask> location.",
#     "A sentence with <mask> placed <mask> differently.",
#     "Here is one more <mask> example with <mask> tokens.",
#     "Testing with <mask> and another <mask> in the text.",
#     "The <mask> example includes multiple <mask> usages.",
#     "This <mask> sample has several <mask> tokens inside.",
#     "An input text <mask> with various <mask> places.",
#     "Final example of <mask> text with masked <mask> tokens.",
# ]
#
# ground_truth_tokens_list = [
#     ["a", "b"],
#     ["c", "d"],
#     ["e", "f"],
#     ["g", "h"],
#     ["token", "tokens"],
#     ["token", "token"],
#     ["token", "token"],
#     ["token", "tokens"],
#     ["token", "token"],
#     ["token", "token"]
# ]

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
predictor = MaskPredictor()
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
