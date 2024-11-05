import torch
from torch.utils.data import Dataset, DataLoader, random_split

# Define custom dataset
class TextDataset(Dataset):
    def __init__(self, masked_text_list, ground_truth_tokens_list):
        self.X = masked_text_list
        self.Y = ground_truth_tokens_list

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# Assuming X and Y are already defined as masked_text_list and ground_truth_tokens_list
dataset = TextDataset(X, Y)

# Split the dataset into train, validation, and test sets
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders
batch_size = 32  # Choose batch size according to your preference
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)