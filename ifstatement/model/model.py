import time
import torch
import torch.nn as nn
import torch.optim as optim

from transformers import RobertaTokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader, random_split

from tqdm import tqdm


TOKENIZER = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
MODEL = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')


class MaskPredictorModel(nn.Module):
    def __init__(self):
        super(MaskPredictorModel, self).__init__()
        self.tokenizer = TOKENIZER
        self.model = MODEL

    def forward(self, inputs):
        # inputs = self.prepare_inputs(input_text)
        outputs = self.model.generate(
            inputs,  # inputs['input_ids'],
            max_new_tokens=1,  # Adjust max_length based on expected output
            num_return_sequences=1
        )
        return outputs

    @staticmethod
    def calculate_match(list1: torch.Tensor, list2: torch.Tensor):
        # Ensure both tensors are the same length
        if list1.shape != list2.shape:
            raise ValueError("Both tensors must have the same shape.")

        # Calculate the number of matches
        matches = (list1 == list2).sum().item()

        # Calculate precision
        # precision = matches / list1.numel()  # total number of elements
        # print(f"{matches} / {list1.numel()} = {matches / list1.numel()}")
        return matches

    def train_model(self, data_loader, epochs=3, lr=5e-5):
        # Define the optimizer and scheduler
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        num_training_steps = epochs * len(data_loader)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                    num_training_steps=num_training_steps)

        self.model.train()

        t0 = time.time()
        t_tmp = t0
        t1 = t0
        for epoch in range(epochs):
            # print(f"Epoch {epoch + 1}/{epochs}")
            # t_tmp = time.time()
            total_loss = 0
            precision_val = 0.0
            precision_cnt = 0
            batch_prediction_list = []
            batch_truth_list = []

            for idx_batch, batch in enumerate(data_loader):
                input_tensor, target_tensor, input_raw, target_raw = batch
                input_tensor = input_tensor.to(self.model.device)
                target_tensor = target_tensor.to(self.model.device)
                # decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(target_tensor)
                # print("*" * 80)
                # print(f"target_tensor: {target_tensor[0][:20]}")
                # print(f"decoder_input_ids: {decoder_input_ids[0][:20]}")
                # print("*" * 80)

                outputs = self.model(input_tensor, decoder_input_ids=target_tensor, labels=target_tensor) # decoder_input_ids
                with torch.no_grad():
                    max_len = target_tensor.size(1)
                    generated_ids = self.model.generate(
                        input_tensor,
                        max_new_tokens=max_len,
                        num_return_sequences=1
                    )
                    # generated_text = [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
                    # batch_prediction_list += generated_text
                    # truth_text = [self.tokenizer.decode(g, skip_special_tokens=True) for g in target_tensor]
                    # batch_truth_list += truth_text
                    # print(f"generated_ids: {generated_ids.shape}, target_tensor: {target_tensor.shape}")


                    # for gen_id in generated_ids:
                    #     decoded_tokens = [self.tokenizer.decode([token_id], skip_special_tokens=True) for token_id in
                    #                       gen_id]
                    #     batch_prediction_list.extend(decoded_tokens)  # xxxx
                    #
                    #     # Convert ground truth target tensor to tokens and add to batch_truth_list
                    # for truth_id in target_tensor:
                    #     decoded_truth_tokens = [self.tokenizer.decode([token_id], skip_special_tokens=True) for token_id
                    #                             in truth_id]
                    #     batch_truth_list.extend(decoded_truth_tokens)  # xxxx
                    for idx_element, gen_id, truth_id, one_input_raw, one_target_raw in zip(range(len(generated_ids)), generated_ids, target_tensor, input_raw, target_raw):
                        # Truncate/pad the generated and truth tokens to match lengths
                        gen_tokens = [self.tokenizer.decode([token_id], skip_special_tokens=True) for token_id in
                                      gen_id[:max_len]]
                        truth_tokens = [self.tokenizer.decode([token_id], skip_special_tokens=True) for token_id in
                                        truth_id[:max_len]]
                        # print(f"batch {idx_batch} element {idx_element + 1:03d}/{len(generated_ids):06d} gen_id [{len(gen_id)}]: {gen_id}")
                        # print(f"batch {idx_batch} element {idx_element + 1:03d}/{len(generated_ids):06d} gen_tokens [{len(gen_tokens)}]: {gen_tokens}")
                        # print(f"batch {idx_batch} element {idx_element + 1:03d}/{len(generated_ids):06d} truth_id [{len(truth_id)}]: {truth_id}")
                        # print(f"batch {idx_batch} element {idx_element + 1:03d}/{len(generated_ids):06d} truth_tokens [{len(truth_tokens)}]: {truth_tokens}")
                        # print(f"batch {idx_batch} element {idx_element + 1:03d}/{len(generated_ids):06d} one_input_raw [{len(one_input_raw)}]: {one_input_raw}")
                        # print(f"batch {idx_batch} element {idx_element + 1:03d}/{len(generated_ids):06d} one_target_raw [{len(one_target_raw)}]: {one_target_raw}")
                        # print("*" * 200)

                        batch_prediction_list.extend(gen_tokens)
                        batch_truth_list.extend(truth_tokens)

                        precision_val += self.calculate_match(gen_id[:max_len], truth_id[:max_len])
                        precision_cnt += len(gen_id)

                loss = outputs.loss
                # print(f"batch {idx_batch}: loss = {loss.item():.6f}")

                # Backpropagation
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                total_loss += loss.item()
            t_tmp = time.time()
            time_cost = t_tmp - t1
            time_total = t_tmp - t0
            time_remain = time_total / (epoch + 1) * (epochs - epoch - 1)

            print(f"[Epoch {epoch + 1:04d}/{epochs:04d}] Loss: {total_loss / len(data_loader):.6f} "
                  f"lr: {optimizer.param_groups[0]['lr']:.6f} precision: {precision_val/precision_cnt:.6f} "
                  f"(t_cost: {time_cost/60:.2f} min, t_total: {time_total/60:.2f} min, t_remain: {time_remain/60:.2f} min)")
            t1 = t_tmp
            # print("=" * 200)
            # print(f"Pred [{len(batch_prediction_list)}]: {batch_prediction_list}")
            # print(f"True [{len(batch_truth_list)}]: {batch_truth_list}")
            # print("=" * 200)


# Define custom dataset
class TextDataset(Dataset):
    def __init__(self, masked_text_list, ground_truth_tokens_list):
        self.tokenizer = TOKENIZER  # RobertaTokenizer.from_pretrained('Salesforce/codet5-base')

        print(f"Example masked_text_list[0]: {masked_text_list[0]}")
        print(f"Example ground_truth_tokens_list[0]: {ground_truth_tokens_list[0]}")
        # Process input text by replacing "<mask>" with "<extra_id_0>", "<extra_id_1>", etc.
        input_text = []
        for item in tqdm(masked_text_list):
            count = 0
            while "<mask>" in item:
                item = item.replace("<mask>", f"<extra_id_{count}>", 1)
                count += 1
            input_text.append(item)

        # Tokenize input text
        self.X = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).input_ids
        self.X_raw = input_text
        print(f"self.X: {self.X.shape}: {self.X}")
        print(f"self.X_raw[0]: {len(self.X_raw)} | {len(self.X_raw[0])} | {self.X_raw[0]}")

        # Process ground truth tokens with padding
        max_len = max(len(tokens) for tokens in ground_truth_tokens_list)  # Find max length in ground truth
        print(f"max_len: {max_len}")
        padded_ground_truths = []

        # for tokens in tqdm(ground_truth_tokens_list):
        #     # Convert tokens to IDs and pad to max length
        #     token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        #     # print(f"tokens: {tokens}, token_ids: {token_ids}")
        #     padded_token_ids = token_ids + [self.tokenizer.pad_token_id] * (max_len - len(token_ids))
        #     padded_ground_truths.append(padded_token_ids)
        # for tokens in tqdm(ground_truth_tokens_list):
        #     # Convert tokens to IDs and pad to max length
        #     # Convert tokens to single token IDs instead of full strings
        #     token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        #     token_ids = [self.tokenizer.convert_tokens_to_ids(token) if isinstance(token, str) else token for token in
        #                  tokens]  # xxxx
        #     # print(f"tokens: {tokens}, token_ids: {token_ids}")
        #     padded_token_ids = token_ids + [self.tokenizer.pad_token_id] * (max_len - len(token_ids))
        #     padded_token_ids = token_ids + [self.tokenizer.pad_token_id] * (max_len - len(token_ids))
        #     padded_ground_truths.append(padded_token_ids)
        #     padded_ground_truths.append(padded_token_ids)
        for tokens in tqdm(ground_truth_tokens_list):
            token_ids = [self.tokenizer.convert_tokens_to_ids(token) if isinstance(token, str) else token for token in
                         tokens]
            padded_token_ids = token_ids + [self.tokenizer.pad_token_id] * (max_len - len(token_ids))
            padded_ground_truths.append(padded_token_ids)

        self.Y = torch.tensor(padded_ground_truths)
        self.Y_raw = ground_truth_tokens_list
        print(f"self.Y: {self.Y.shape}: {self.Y}")
        print(f"self.Y_raw[0]: {len(self.Y_raw)} | {len(self.Y_raw[0])} | {self.Y_raw[0]}")
        # assert len(self.X) == len(self.Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.X_raw[idx], self.Y_raw[idx]


#
# import torch
# import torch.nn as nn
# import torch.optim as optim
#
# from transformers import RobertaTokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup
# from torch.utils.data import Dataset, DataLoader, random_split
#
# from tqdm import tqdm
#
#
# TOKENIZER = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
# MODEL = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')
#
#
# class MaskPredictorModel(nn.Module):
#     def __init__(self, model_name='Salesforce/codet5-base'):
#         super(MaskPredictorModel, self).__init__()
#
#         self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
#         self.model = T5ForConditionalGeneration.from_pretrained(model_name)
#
#     # def prepare_inputs(self, input_text):
#     #     # Replace <mask> with the tokenizer's mask token
#     #     input_text = input_text.replace("<mask>", self.tokenizer.mask_token)
#     #     inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
#     #     return inputs
#
#     def forward(self, inputs):
#         # inputs = self.prepare_inputs(input_text)
#         outputs = self.model.generate(
#             inputs,  # inputs['input_ids'],
#             max_new_tokens=5,  # Adjust max_length based on expected output
#             num_return_sequences=1
#         )
#         return outputs
#
#     def train_model(self, data_loader, epochs=3, lr=5e-5):
#         # Define the optimizer and scheduler
#         optimizer = optim.AdamW(self.model.parameters(), lr=lr)
#         num_training_steps = epochs * len(data_loader)
#         scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
#                                                     num_training_steps=num_training_steps)
#
#         self.model.train()
#
#         for epoch in range(epochs):
#             # print(f"Epoch {epoch + 1}/{epochs}")
#             total_loss = 0
#             batch_prediction_list = []
#             batch_truth_list = []
#
#             for idx_batch, batch in enumerate(data_loader):
#                 input_tensor, target_tensor = batch
#                 input_tensor = input_tensor.to(self.model.device)
#                 target_tensor = target_tensor.to(self.model.device)
#                 # print(f"[input_tensor] {input_tensor.shape}")
#                 # print(f"[target_tensor] {target_tensor.shape}")
#
#                 # Flatten target_texts into strings for training
#                 # target_texts_flat = [" ".join(target) for target in target_texts]
#
#                 # Tokenize inputs and targets with consistent padding and max_length
#                 # inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=50)
#                 # targets = self.tokenizer(target_texts_flat, return_tensors="pt", padding=True, truncation=True,
#                 #                          max_length=50).input_ids
#
#                 # Move tensors to the appropriate device if using GPU
#                 # inputs = {key: val.to(self.model.device) for key, val in inputs.items()}
#                 # targets = targets.to(self.model.device)
#
#                 # Create `decoder_input_ids` from target tokens for T5
#                 decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(target_tensor)
#
#                 # Forward pass with labels for training
#                 # print(f"targets: {targets.shape}, {targets}")
#                 # print(f"inputs['input_ids']: {inputs['input_ids'].shape}, {inputs['input_ids']}")
#                 # print(f"decoder_input_ids: {decoder_input_ids.shape}, {decoder_input_ids}")
#                 outputs = self.model(input_tensor, decoder_input_ids=decoder_input_ids, labels=target_tensor)
#                 with torch.no_grad():
#                     generated_ids = self.model.generate(
#                         input_tensor,
#                         max_new_tokens=5,
#                         num_return_sequences=1
#                     )
#                     generated_text = [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
#                     batch_prediction_list += generated_text
#                     truth_text = [self.tokenizer.decode(g, skip_special_tokens=True) for g in target_tensor]
#                     batch_truth_list += truth_text
#                     # print(f"Predicted text: {generated_text}")
#                 # generated_text = self.tokenizer.decode(outputs, skip_special_tokens=True)
#                 # print(f"Predicted text: {generated_text}")
#
#                 loss = outputs.loss
#                 print(f"batch {idx_batch}: loss = {loss.item():.6f}")
#
#                 # Backpropagation
#                 loss.backward()
#                 optimizer.step()
#                 scheduler.step()
#                 optimizer.zero_grad()
#
#                 total_loss += loss.item()
#
#             print(f"Epoch {epoch + 1}/{epochs}: Loss: {total_loss / len(data_loader)} lr: {optimizer.param_groups[0]['lr']:.9f}")
#             print(f"Pred: {batch_prediction_list}")
#             print(f"True: {batch_truth_list}")
#
#
# # Define custom dataset
# class TextDataset(Dataset):
#     def __init__(self, masked_text_list, ground_truth_tokens_list):
#         self.tokenizer = TOKENIZER  # RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
#
#         print(f"Example masked_text_list[0]: {masked_text_list[0]}")
#         print(f"Example ground_truth_tokens_list[0]: {ground_truth_tokens_list[0]}")
#         # Process input text by replacing "<mask>" with "<extra_id_0>", "<extra_id_1>", etc.
#         input_text = []
#         for item in tqdm(masked_text_list):
#             count = 0
#             while "<mask>" in item:
#                 item = item.replace("<mask>", f"<extra_id_{count}>", 1)
#                 count += 1
#             input_text.append(item)
#
#         # Tokenize input text
#         self.X = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).input_ids
#         print(f"self.X: {self.X.shape}: {self.X}")
#
#         # Process ground truth tokens with padding
#         max_len = max(len(tokens) for tokens in ground_truth_tokens_list)  # Find max length in ground truth
#         print(f"max_len: {max_len}")
#         padded_ground_truths = []
#
#         for tokens in tqdm(ground_truth_tokens_list):
#             # Convert tokens to IDs and pad to max length
#             token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
#             # print(f"tokens: {tokens}, token_ids: {token_ids}")
#             padded_token_ids = token_ids + [self.tokenizer.pad_token_id] * (max_len - len(token_ids))
#             padded_ground_truths.append(padded_token_ids)
#
#         self.Y = torch.tensor(padded_ground_truths)
#         print(f"self.Y: {self.Y.shape}: {self.Y}")
#         # assert len(self.X) == len(self.Y)
#
#     # def prepare_inputs(self, input_text):
#     #     # Replace <mask> with the tokenizer's mask token
#     #     input_text = input_text.replace("<mask>", self.tokenizer.mask_token)
#     #     inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
#     #     return inputs
#
#     def __len__(self):
#         return len(self.X)
#
#     def __getitem__(self, idx):
#         return self.X[idx], self.Y[idx]