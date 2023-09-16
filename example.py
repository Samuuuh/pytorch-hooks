import numpy as np

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from collections import OrderedDict 

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

# Set a seed
seed_value = 0
np.random.seed(seed_value)
torch.manual_seed(seed_value)

class NewModel(nn.Module):
    def __init__(self, output_layers, *args):
        super().__init__(*args)
        self.output_layers = output_layers
        
        # Pretrained Model
        self.pretrained = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Where the output of the hooks will be stored
        self.selected_out = OrderedDict()

        # Register the forward hook on the given output layers
        # Forward Hook is triggered every time after the method foward of the Pytorch AutoGrad Function grad_fn
        # We can modify the output by returning the modified output from the hook. 
        # Using forward_pre_hook the user can modify the input but returning the modified input value as a tuple or just a single modified value in the hook.
        self.fhooks = []
        for i,l in enumerate(list(self.pretrained._modules.keys())):
            if i in self.output_layers:
                self.fhooks.append(getattr(self.pretrained,l).register_forward_hook(self.custom_forward_hook(l)))
        
        self.bhooks = []


    def custom_forward_hook(self, layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output

        return hook
    

    def forward(self, x):
        out = self.pretrained(x)
        return out, self.selected_out
    
# 7 -> layer4
# 8 -> avgpool
model = NewModel(output_layers = [7,8])

# Random array with the dimension of the layer 4
target_ft = torch.rand((2048,8,8))

learning_rate = 0.00001
params = [p for p in model.parameters() if p.requires_grad]
optimizer = Adam(params, lr=learning_rate)
lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.9)

# Define the Image
batch_size = 1
channels = 3
height = 256
width = 256

image_array = np.random.randint(0, 256, size=(batch_size, channels, height, width)).astype('float32')
x = torch.from_numpy(image_array)

out, layerout = model(x)
layer4out = layerout['layer4']