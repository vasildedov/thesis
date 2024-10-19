from datasetsforecast.m4 import M4, M4Evaluation, M4Info
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train_valid_split(group):
    df, *_ = M4.load(directory='data', group=group)
    df['ds'] = df['ds'].astype('int')
    horizon = M4Info[group].horizon
    valid = df.groupby('unique_id').tail(horizon)
    train = df.drop(valid.index)
    return train, valid

hourly_train, hourly_valid = train_valid_split('Hourly')

# Custom PyTorch Dataset for hourly_train
class HourlyDataset(Dataset):
    def __init__(self, dataframe, seq_length):
        """
        Args:
            dataframe (DataFrame): The pandas dataframe containing unique_id, ds, and y.
            seq_length (int): The length of the input sequence.
        """
        self.data = dataframe
        self.seq_length = seq_length

        # Group the dataframe by unique_id to keep each time series separate
        self.groups = {uid: group for uid, group in self.data.groupby('unique_id')}

        # Get a list of unique ids (keys) to index the time series
        self.unique_ids = list(self.groups.keys())

    def __len__(self):
        # Return the number of unique time series
        return len(self.unique_ids)

    def __getitem__(self, idx):
        # Get the time series corresponding to the unique_id at the given index
        unique_id = self.unique_ids[idx]  # Get unique_id by index
        group = self.groups[unique_id]    # Get the group (time series) for the unique_id

        # Extract the 'y' values as the time series
        series = group['y'].values

        # Ensure the series is long enough for the sequence length
        if len(series) < self.seq_length + 1:
            # Pad with zeros if too short
            series = torch.cat([torch.zeros(self.seq_length + 1 - len(series)), torch.tensor(series)])

        inputs = torch.tensor(series[:self.seq_length], dtype=torch.float32)
        target = torch.tensor(series[self.seq_length], dtype=torch.float32)

        return inputs, target

# Define the sequence length (number of previous time steps used as input)
seq_length = 2

# Create dataset
dataset = HourlyDataset(dataframe=hourly_train, seq_length=seq_length)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Example loop to load data
for batch_inputs, batch_targets in dataloader:
    print("Input batch:", batch_inputs)
    print("Target batch:", batch_targets)


# Example PyTorch model (e.g., LSTM for time series forecasting)
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        # Take the last output of the sequence for the forecast
        out = self.fc(lstm_out[:, -1, :])
        return out

# Hyperparameters
input_size = 1         # Input features (e.g., 'y' values from the time series)
hidden_size = 64       # LSTM hidden layer size
output_size = 1        # Single value prediction (regression)
num_layers = 1         # Number of LSTM layers
seq_length = 2         # As defined in your dataset
batch_size = 1         # Batch size


# Initialize model, loss function, and optimizer
model = SimpleLSTM(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
criterion = nn.MSELoss()  # Mean Squared Error for regression tasks
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example DataLoader (from the dataset we previously defined)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for batch_inputs, batch_targets in dataloader:
        # Reshape inputs for LSTM (batch_size, seq_length, input_size)
        batch_inputs = batch_inputs.unsqueeze(-1)  # Add the input size dimension
        batch_targets = batch_targets.unsqueeze(-1)  # Make targets 2D to match the output size

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_inputs)

        # Compute the loss
        loss = criterion(outputs, batch_targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Example of using the DataLoader for predictions (no training)
model.eval()  # Set the model to evaluation mode (no gradients needed)
with torch.no_grad():
    for batch_inputs, batch_targets in dataloader:
        batch_inputs = batch_inputs.unsqueeze(-1)
        outputs = model(batch_inputs)
        print("Predictions:", outputs)
        print("Actual:", batch_targets)

