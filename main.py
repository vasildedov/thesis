from datasetsforecast.m4 import M4, M4Evaluation, M4Info

def train_valid_split(group):
    df, *_ = M4.load(directory='data', group=group)
    df['ds'] = df['ds'].astype('int')
    horizon = M4Info[group].horizon
    valid = df.groupby('unique_id').tail(horizon)
    train = df.drop(valid.index)
    return train, valid

hourly_train, hourly_valid = train_valid_split('Hourly')

import torch
from torch.utils.data import Dataset, DataLoader

class M4TimeSeriesDataset(Dataset):
    def __init__(self, dataframe):
        """
        Initialize the dataset with the dataframe containing the time series data.
        """
        self.data = dataframe
        self.groups = dataframe.groupby('unique_id')
        self.series_list = list(self.groups)

    def __len__(self):
        return len(self.series_list)

    def __getitem__(self, idx):
        """
        Return the full time series for a specific unique_id as a PyTorch tensor.
        """
        unique_id, series_data = self.series_list[idx]
        values = torch.tensor(series_data['y'].values, dtype=torch.float32)
        return values, unique_id


import torch.nn as nn


class SimpleRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=20, num_layers=1, output_size=1):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        out = self.fc(rnn_out)
        return out.squeeze(-1)  # Remove last dimension for compatibility

import torch.optim as optim
device = torch.device("cuda:0")


def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=10):
    model.train()
    train_losses, valid_losses = [], []

    for epoch in range(num_epochs):
        running_loss = 0.0

        for inputs, _ in train_loader:
            inputs = inputs.unsqueeze(-1).to(device)  # Add feature dimension
            optimizer.zero_grad()
            outputs = model(inputs)

            # Predict the next value for each time step
            targets = inputs[:, 1:, 0]  # Shifted target sequence
            loss = criterion(outputs[:, :-1], targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, _ in valid_loader:
                inputs = inputs.unsqueeze(-1).to(device)
                outputs = model(inputs)
                targets = inputs[:, 1:, 0]
                loss = criterion(outputs[:, :-1], targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(valid_loader)
        valid_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        model.train()

    return train_losses, valid_losses

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model parameters
input_size = 1
hidden_size = 20
output_size = 1
num_layers = 1
num_epochs = 10

# Instantiate model, criterion, and optimizer
model = SimpleRNN(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# DataLoaders
train_loader = DataLoader(M4TimeSeriesDataset(hourly_train), batch_size=1, shuffle=True)
valid_loader = DataLoader(M4TimeSeriesDataset(hourly_valid), batch_size=1, shuffle=False)

# Train the model
train_losses, valid_losses = train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs)

import matplotlib.pyplot as plt

# Plot training and validation loss
plt.plot(train_losses, label='Train Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Plot predictions vs actuals on validation set
model.eval()
with torch.no_grad():
    for i, (inputs, _) in enumerate(valid_loader):
        inputs = inputs.unsqueeze(-1).to(device)
        outputs = model(inputs)

        # Plot first validation series
        if i == 0:
            plt.plot(inputs.squeeze().cpu().numpy(), label='Actual')
            plt.plot(outputs.squeeze().cpu().numpy(), label='Predicted')
            plt.xlabel("Time Step")
            plt.ylabel("Value")
            plt.legend()
            plt.show()
            break
