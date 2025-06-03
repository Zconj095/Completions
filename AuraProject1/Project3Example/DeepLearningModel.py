import torch
import torch.nn as nn
import numpy as np
import cupy as cp

# Define a simple LSTM model
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTMPredictor, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size).cuda(),
                            torch.zeros(1,1,self.hidden_layer_size).cuda())

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


import numpy as np
# Assuming 'data' is your time series data
data = np.random.rand(100)  # Example: Replace with actual time series data

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)
# Followed by LSTM model initialization and training logic...

# Example: Manually defining 'data' for demonstration purposes
data = np.random.rand(100)  # Generating 100 random data points; replace this with your actual data

# Now 'data' is defined and can be used to create sequences
seq_length = 5  # Example sequence length
x, y = create_sequences(data, seq_length)



import torch
import torch.nn as nn
import numpy as np
import cupy as cp

# Move data to GPU
x_gpu = torch.tensor(x, dtype=torch.float32).cuda()
y_gpu = torch.tensor(y, dtype=torch.float32).cuda()

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

