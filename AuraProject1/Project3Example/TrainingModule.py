import torch
import torch.nn as nn
import numpy as np
import cupy as cp
import numpy as np
import torch.nn as nn

# Example: Generating synthetic sequential data
# Let's assume we're generating synthetic time series data for demonstration
time_series_length = 1000
time_series = np.sin(np.linspace(0, 40 * np.pi, time_series_length))

class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTMPredictor, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        
        # The linear layer that maps from hidden state space to output space
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_layer_size]
        # shape of self.hidden_cell: (h_n, c_n), both with shape (num_layers, batch_size, hidden_layer_size)
        lstm_out, _ = self.lstm(input_seq)
        
        # Only take the output from the final timestep
        predictions = self.linear(lstm_out[-1].view(input_seq.size(1), -1))
        return predictions

    def init_hidden(self, batch_size):
        # Initializes hidden state
        # This method is not strictly necessary if you're resetting the hidden state at each batch iteration
        # But it's useful if you want to clear the hidden state manually at any point
        return (torch.zeros(1, batch_size, self.hidden_layer_size),
                torch.zeros(1, batch_size, self.hidden_layer_size))

# Function to create sequences suitable for LSTM input
def create_sequences(data, sequence_length):
    xs, ys = [], []
    for i in range(len(data) - sequence_length):
        x = data[i:(i + sequence_length)]
        y = data[i + sequence_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Generating sequences
sequence_length = 10  # Length of the sequence to be fed into the LSTM
x, y = create_sequences(time_series, sequence_length)

# Convert to tensors and move to GPU
x_gpu = torch.tensor(x, dtype=torch.float32).cuda().unsqueeze(-1)  # Adding an extra dimension for LSTM input
y_gpu = torch.tensor(y, dtype=torch.float32).cuda().unsqueeze(-1)  # Target tensor also needs to match input dimensions

# Initialize the model
model = LSTMPredictor(input_size=1, hidden_layer_size=100, output_size=1).cuda()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



# Training loop
epochs = 150
for i in range(epochs):
    for seq, labels in zip(x_gpu, y_gpu):
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).cuda(),
                             torch.zeros(1, 1, model.hidden_layer_size).cuda())
        
        y_pred = model(seq)
        
        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if i%25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
