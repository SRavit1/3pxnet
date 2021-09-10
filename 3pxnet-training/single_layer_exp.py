import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import onnx

import binarized_modules

class Net(nn.Module):
    def __init__(self, binary, padding, pooling):
        super(Net, self).__init__()
        
        self.binary = binary
        self.padding = padding
        self.pooling = pooling

        self.conv0 = binarized_modules.BinarizeConv2d(1, 1, 1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        if self.binary:
            self.conv1 = binarized_modules.BinarizeConv2d(1, 1, 32, 32, kernel_size=3, stride=1, padding=padding, bias=False)
        else:
            self.conv1 = binarized_modules.TernarizeConv2d(0.9, 1, 1, 32, 32, kernel_size=3, stride=1, padding=padding, bias=False)
        self.conv2 = binarized_modules.BinarizeConv2d(1, 1, 32, 1, kernel_size=3, stride=1, padding=1, bias=False)

        if self.pooling:
            self.pool1 = nn.MaxPool2d((3, 3))

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        if self.pooling:
            x = self.pool1(x)
        x = self.conv2(x)

        return x


for w in range(2): # precision
    for x in range(2): # input size
        for y in range(2): # padding
            for z in range(2): # pooling
                id = str(w) + str(x) + str(y) + str(z)
                print("id is", id)

                model = Net(w, y, z)
                dim = 16 if x else 8
                x_in = torch.randn(1, 1, dim, dim, requires_grad=True).to('cpu')

                torch.onnx.export(model, x_in, "training_data/single_exp_" + id + ".onnx", verbose=True, opset_version=9, input_names = ['input'], output_names = ['output'])
