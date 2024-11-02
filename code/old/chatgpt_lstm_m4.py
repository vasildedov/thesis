from datasetsforecast.m4 import M4, M4Evaluation, M4Info
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

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
    def __init__(self, dataframe):
        """
        Args:
            dataframe (DataFrame): The pandas dataframe containing unique_id, ds, and y.
        """
        self.data = dataframe
        self.groups = dataframe.groupby('unique_id')

        # Store the unique ids and the indices for each group
        self.group_indices = {uid: group.index.tolist() for uid, group in self.groups}

    def __len__(self):
        # Return the total number of groups (unique time series)
        return len(self.group_indices)

    def __getitem__(self, idx):
        """
        Instead of returning a single row, return all rows for a given unique_id group.
        idx is an index into the unique_ids, not the dataframe rows.
        """
        unique_id = list(self.group_indices.keys())[idx]  # Get the unique_id for this batch
        indices = self.group_indices[unique_id]  # Get all row indices for this unique_id
        group_data = self.data.iloc[indices]  # Get the data for this unique_id

        # Return the 'y' values as inputs and targets along with the unique_id
        inputs = torch.tensor(group_data['y'].values, dtype=torch.float32)
        return inputs, unique_id


# Custom collate function to handle grouping
def collate_fn(batch):
    """
    This collate function ensures that each batch consists of sequences from a single unique_id.
    It receives a batch, which is a list of tuples (inputs, unique_id).
    """
    inputs = [item[0] for item in batch]  # Extract inputs
    unique_ids = [item[1] for item in batch]  # Extract unique_ids

    # Stack inputs into a batch and return unique_id as well
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)  # Pad to handle different lengths
    return inputs, unique_ids

# Define the sequence length (number of previous time steps used as input)
seq_length = 2

# Create dataset
dataset = HourlyDataset(dataframe=hourly_train)

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
batch_size = 4         # Batch size


# Initialize model, loss function, and optimizer
model = SimpleLSTM(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
criterion = nn.MSELoss()  # Mean Squared Error for regression tasks
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example DataLoader (from the dataset we previously defined)
dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

# check data
for batch_inputs, batch_targets in dataloader:
    print("Input batch:", batch_inputs)
    print("Target batch:", batch_targets)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    losses = []
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

        # add loss to list
        losses.append(loss)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.mean():.4f}')

# Example of using the DataLoader for predictions (no training)
model.eval()  # Set the model to evaluation mode (no gradients needed)
with torch.no_grad():
    for batch_inputs, batch_targets in dataloader:
        batch_inputs = batch_inputs.unsqueeze(-1)
        outputs = model(batch_inputs)
        print("Predictions:", outputs)
        print("Actual:", batch_targets)

torch.cuda.is_available()

