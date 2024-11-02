import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# make a sine wave
N = 1000
series = np.sin(0.1*np.arange(N))

plt.plot(series)
plt.show()

# use past T values to predict the next value
T = 10
X = []
Y = []
for t in range(len(series)-T):
    x = series[t:t+T]
    X.append(x)
    y = series[t+T]
    Y.append(y)

X = np.array(X).reshape(-1, T, 1)
Y = np.array(Y).reshape(-1, 1)

N = len(X)

X.shape, Y.shape

device = torch.device("cuda:0")

class SimpleRNN(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_rnnlayers, n_outputs):
        super(SimpleRNN, self).__init__()
        self.D = n_inputs
        self.M = n_hidden
        self.K = n_outputs
        self.L = n_rnnlayers

        self.rnn = nn.RNN(
            input_size=self.D,
            hidden_size=self.M,
            num_layers=self.L,
            nonlinearity='relu',
            batch_first=True  # convention shape is samples x sequence x features
        )
        self.fc = nn.Linear(self.M, self.K)

    def forward(self, X):
        h0 = torch.zeros(self.L, X.size(0), self.M).to(device)  # initial hidden state

        # get RNN unit output - (N, T, M) and hidden state at each layer
        # currently hidden state is not needed
        out, _ = self.rnn(X, h0)

        # we only want (h(T)) at the final time step
        out = self.fc(out[:, -1, :])
        return out


model = SimpleRNN(1, 5, 1, 1)
model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# make inputs and targets
X_train = torch.from_numpy(X[:-N//2].astype(np.float32))
y_train = torch.from_numpy(Y[:-N//2].astype(np.float32))
X_test = torch.from_numpy(X[:-N//2].astype(np.float32))
y_test = torch.from_numpy(Y[:-N//2].astype(np.float32))

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

def full_gd(model, criterion, optimizer, X_train, y_train, X_test, y_test, epochs=200):
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)

    for it in range(epochs):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # backward and optimize
        loss.backward()
        optimizer.step()

        # save losses
        train_losses[it] = loss.item()

        # test loss
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        test_losses[it] = test_loss.item()

        if (it+1) % 5 == 0:
            print(f'Epoch {it+1}/{epochs}, Train Loss: {loss.item(): .4f},'
                  f'Test Loss: {test_loss.item(): .4f}')

    return train_losses, test_losses


train_losses, test_losses = full_gd(model, criterion, optimizer, X_train, y_train, X_test, y_test)

# plot losses per iteration
plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='test loss')
plt.legend()
plt.show()

# forecast future values using only self-predictions for future predictions
validation_target = Y[-N//2:]
validation_predictions = []

last_x = X_test[0].view(T)

while len(validation_predictions) < len(validation_target):
    input_ = last_x.reshape(1, T, 1)
    p = model(input_)
    # [0,0] -> scalar

    validation_predictions.append(p[0,0].item())

    last_x = torch.cat((last_x[1:], p[0]))

plt.plot(validation_target, label='forecast target')
plt.plot(validation_predictions, label='forecast prediction')
plt.legend()
