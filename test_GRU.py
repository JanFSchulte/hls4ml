import torch
from torch import nn
import random, os
import numpy as np

from hls4ml.converters import convert_from_pytorch_model
from hls4ml.utils.config import config_from_pytorch_model

seed = 2020
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(GRUModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

        # GRU layers
        self.gru = nn.GRU(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, 10, self.hidden_dim).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out


class GRUNet(nn.Module):
    def __init__(self):
        super(GRUNet, self).__init__()
        self.rnn = nn.GRU(10, 20, num_layers=1, batch_first=True)
        #self.fc = nn.Linear(20, 20)

    def forward(self, x, h0):
        output, hnn = self.rnn(x, h0)
        #output = self.fc(output)
        return output
    
input = torch.randn(1, 1, 10)
input = torch.tensor([[[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]]])
h0 = torch.zeros(1, 1, 20)

model = GRUNet()



pytorch_prediction = model(input, h0).detach().numpy()

config = config_from_pytorch_model(model, inputs_channel_last=True,transpose_outputs=False)

output_dir = "test_pytorch"
backend = "Vivado"
io_type = 'io_parallel'

hls_model = convert_from_pytorch_model(
    model,
    (None, 1, 10),
    hls_config=config,
    output_dir=output_dir,
    backend=backend,
    io_type=io_type,
)
hls_model.compile()

print (pytorch_prediction.size)
hls_prediction = np.reshape(hls_model.predict(input.detach().numpy()), (1, 1, 20))
#hls_prediction = np.transpose(np.reshape(hls_model.predict(X_input), (1, out_height, out_width, n_out)), (0, 3, 1, 2))


print (pytorch_prediction)
print (hls_prediction)