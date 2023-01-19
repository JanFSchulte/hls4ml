import torch
import torch.nn as nn
import torch.nn.functional as F

from hls4ml.converters.pytorch_to_hls import pytorch_to_hls
from hls4ml.converters import convert_from_pytorch_model
from hls4ml.converters import convert_from_torchscript_model
from hls4ml.utils.config import config_from_pytorch_model

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = Net()

class LayerLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        # Instead of our custom parameters, we use a Linear layer with single input and single output
        self.linear = nn.Linear(1, 1)
                
    def forward(self, x):
        # Now it only takes a call to the layer to make predictions
        return self.linear(x)

model = LayerLinearRegression()

script_model = torch.jit.script(model)

#print(script_model.code)
#for module in script_model.modules():
#    print("new module")
#    print(module)
print (script_model.original_name)
for layer_name, pytorch_layer in script_model.named_modules():
    print(layer_name)
    print(pytorch_layer.original_name)

#config = config_from_pytorch_model(model)
#config["PytorchModel"] = model
#config["InputShape"] = [1]
#hls_model = convert_from_pytorch_model(model, (1,1), hls_config = config )

config = config_from_pytorch_model(script_model)
hls_model = convert_from_torchscript_model(script_model, (1,1), hls_config = config )

