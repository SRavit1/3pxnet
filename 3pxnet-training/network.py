import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import utils
import utils_own

import binarized_modules
import binarized_modules_multi
import uuid

def getModelFilename(model, modelName):
  extension = modelName + "_model"
  if model.full:
    extension += "_full"
  else:
    if model.binary:
      extension += "_binary"
    else:
      extension += "_ternary"
      extension += "_" + str(model.conv_thres)
  extension += "_bw" + str(model.bitwidth)
  extension += "_input_bw" + str(model.input_bitwidth)
  extension += "_" + model.id
  return extension

class pnet(nn.Module):
    def __init__(self, full=False, binary=True, conv_thres=0.7, align=True, bitwidth=1, input_bitwidth=1, binarize_input=True):
        super(pnet, self).__init__()
        self.align = align
        self.pruned = False

        self.full = full
        self.binary = binary
        self.bitwidth = bitwidth
        self.input_bitwidth = input_bitwidth
        self.binarize_input = binarize_input
        self.conv_thres = conv_thres
        self.id = str(uuid.uuid4())[:8]
        self.name = "pnet"
        self.filename = getModelFilename(self, self.name)

        if full:
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=(0, 0), bias=False)
            self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=(0, 0), bias=False)
            self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=(0, 0), bias=False)
            self.conv4 = nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=(0, 0), bias=False)
            self.conv5 = nn.Conv2d(32, 4, kernel_size=1, stride=1, padding=(0, 0), bias=False)
        elif binary:
            if self.binarize_input:
              self.conv1 = binarized_modules_multi.BinarizeConv2d(input_bitwidth, bitwidth, 3, 32, kernel_size=3, stride=1, padding=(0, 0), bias=False)
            else:
              self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=(0, 0), bias=False)
            self.conv2 = binarized_modules_multi.BinarizeConv2d(bitwidth, bitwidth, 32, 32, kernel_size=3, stride=1, padding=(0, 0), bias=False)
            self.conv3 = binarized_modules_multi.BinarizeConv2d(bitwidth, bitwidth, 32, 32, kernel_size=3, stride=1, padding=(0, 0), bias=False)
            #self.conv4 = binarized_modules_multi.BinarizeConv2d(bitwidth, bitwidth, 32, 2, kernel_size=1, stride=1, padding=(0, 0), bias=False)
            self.conv4 = nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=(0, 0), bias=False)
            #self.conv5 = binarized_modules_multi.BinarizeConv2d(bitwidth, bitwidth, 32, 4, kernel_size=1, stride=1, padding=(0, 0), bias=False)
            self.conv5 = nn.Conv2d(32, 4, kernel_size=1, stride=1, padding=(0, 0), bias=False)
        else:
            if self.binarize_input:
              self.conv1 = binarized_modules_multi.TernarizeConv2d(conv_thres, input_bitwidth, bitwidth, 3, 32, kernel_size=3, stride=1, padding=(0, 0), align=False, bias=False) #cannot be aligned since # input channels < 32
            else:
              self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=(0, 0), bias=False)
            self.conv2 = binarized_modules_multi.TernarizeConv2d(conv_thres, bitwidth, bitwidth, 32, 32, kernel_size=3, stride=1, padding=(0, 0), align=self.align, bias=False)
            self.conv3 = binarized_modules_multi.TernarizeConv2d(conv_thres, bitwidth, bitwidth, 32, 32, kernel_size=3, stride=1, padding=(0, 0), align=self.align, bias=False)
            #self.conv4 = binarized_modules_multi.TernarizeConv2d(conv_thres, bitwidth, bitwidth, 32, 2, kernel_size=1, stride=1, padding=(0, 0), align=self.align, bias=False)
            self.conv4 = nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=(0, 0), bias=False)
            #self.conv5 = binarized_modules_multi.TernarizeConv2d(conv_thres, bitwidth, bitwidth, 32, 4, kernel_size=1, stride=1, padding=(0, 0), align=self.align, bias=False)
            self.conv5 = nn.Conv2d(32, 4, kernel_size=1, stride=1, padding=(0, 0), bias=False)


        self.softmax1 = torch.nn.Softmax(dim = 1)
        self.act = F.relu if self.full else nn.Hardtanh()
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(2)
        self.bn5 = nn.BatchNorm2d(4)
    def forward(self, x):
        pad_width = 0 if x.shape[-1]%2 == 0 else 1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=(pad_width, pad_width))

        x = self.act(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        out1 = self.softmax1(self.bn4(self.conv4(x)))
        out2 = self.bn5(self.conv5(x))

        return out1, out2

class rnet(nn.Module):
    def __init__(self, full=False, binary=True, first_sparsity=0.8, rest_sparsity=0.9, conv_thres=0.7, align=True, bitwidth=1, input_bitwidth=1, binarize_input=True):
        super(rnet, self).__init__()
        self.align = align
        self.pruned = False

        self.full = full
        self.binary = binary
        self.bitwidth = bitwidth
        self.input_bitwidth = input_bitwidth
        self.binarize_input = binarize_input
        self.conv_thres = conv_thres
        self.first_sparsity = first_sparsity
        self.rest_sparsity = rest_sparsity
        self.id = str(uuid.uuid4())[:8]
        self.name = "rnet"
        self.filename = getModelFilename(self, self.name)

        if full:
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=(0, 0), bias=False)
            self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=(0, 0), bias=False)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=(0, 0), bias=False)

            self.fc1 = nn.Linear(64, 128, bias=False)
            self.fc2 = nn.Linear(128, 2, bias=False)
            self.fc3 = nn.Linear(128, 4, bias=False)
        elif binary:
            if self.binarize_input:
              self.conv1 = binarized_modules_multi.BinarizeConv2d(input_bitwidth, bitwidth, 3, 32, kernel_size=3, stride=1, padding=(0, 0), bias=False)
            else:
              self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=(0, 0), bias=False)
            self.conv2 = binarized_modules_multi.BinarizeConv2d(bitwidth, bitwidth, 32, 32, kernel_size=3, stride=1, padding=(0, 0), bias=False)
            self.conv3 = binarized_modules_multi.BinarizeConv2d(bitwidth, bitwidth, 32, 64, kernel_size=3, stride=1, padding=(0, 0), bias=False)

            self.fc1 = binarized_modules_multi.BinarizeLinear(bitwidth, bitwidth, 64, 128, bias=False)
            #self.fc2 = binarized_modules_multi.BinarizeLinear(bitwidth, bitwidth, 128, 2, bias=False)
            #self.fc3 = binarized_modules_multi.BinarizeLinear(bitwidth, bitwidth, 128, 4, bias=False)
            self.fc2 = nn.Linear(128, 2, bias=False)
            self.fc3 = nn.Linear(128, 4, bias=False)
        else:
            if self.binarize_input:
              self.conv1 = binarized_modules_multi.TernarizeConv2d(conv_thres, input_bitwidth, bitwidth, 3, 32, kernel_size=3, stride=1, padding=(0, 0), align=False, bias=False)
            else:
              self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=(0, 0), bias=False)
            self.conv2 = binarized_modules_multi.TernarizeConv2d(conv_thres, bitwidth, bitwidth, 32, 32, kernel_size=3, stride=1, padding=(0, 0), align=self.align, bias=False)
            self.conv3 = binarized_modules_multi.TernarizeConv2d(conv_thres, bitwidth, bitwidth, 32, 64, kernel_size=3, stride=1, padding=(0, 0), align=self.align, bias=False)

            self.fc1 = binarized_modules_multi.TernarizeLinear(first_sparsity, bitwidth, bitwidth, 64, 128, align=self.align, bias=False)
            #self.fc2 = binarized_modules_multi.TernarizeLinear(rest_sparsity, bitwidth, bitwidth, 128, 2, align=self.align, bias=False)
            #self.fc3 = binarized_modules_multi.TernarizeLinear(rest_sparsity, bitwidth, bitwidth, 128, 4, align=self.align, bias=False)
            self.fc2 = nn.Linear(128, 2, bias=False)
            self.fc3 = nn.Linear(128, 4, bias=False)

        self.softmax1 = torch.nn.Softmax(dim = 1)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=(0, 0)) #TODO: Find the padding for "SAME"
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=(0, 0))
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=(0, 0))

        self.act = F.relu if self.full else nn.Hardtanh()

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(2)
        self.bn6 = nn.BatchNorm1d(4)

        
    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.act(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.act(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = torch.reshape(x, (-1, 64))
        x = self.act(self.bn4(self.fc1(x)))
        out1 = self.softmax1(self.bn5(self.fc2(x)))
        out2 = self.bn6(self.fc3(x))

        return out1, out2

class onet(nn.Module):
    def __init__(self, full=False, binary=True, first_sparsity=0.8, rest_sparsity=0.9, conv_thres=0.7, align=True, bitwidth=1, input_bitwidth=1, binarize_input=True):
        super(onet, self).__init__()
        self.align = align
        self.pruned = False

        self.full = full
        self.binary = binary
        self.bitwidth = bitwidth
        self.input_bitwidth = input_bitwidth
        self.binarize_input = binarize_input
        self.conv_thres = conv_thres
        self.first_sparsity = first_sparsity
        self.rest_sparsity = rest_sparsity
        self.id = str(uuid.uuid4())[:8]
        self.name = "onet"
        self.filename = getModelFilename(self, self.name)

        if full:
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=(0, 0), bias=False)
            self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=(0, 0), bias=False)
            self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=(0, 0), bias=False)
            self.conv4 = nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=(0, 0), bias=False)

            self.fc1 = nn.Linear(64, 128, bias=False)
            self.fc2 = nn.Linear(128, 2, bias=False)
            self.fc3 = nn.Linear(128, 4, bias=False)
            self.fc4 = nn.Linear(128, 10, bias=False)
        elif binary:
            if self.binarize_input:
              self.conv1 = binarized_modules_multi.BinarizeConv2d(input_bitwidth, bitwidth, 3, 32, kernel_size=3, stride=1, padding=(0, 0), bias=False)
            else:
              self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=(0, 0), bias=False)
            self.conv2 = binarized_modules_multi.BinarizeConv2d(bitwidth, bitwidth, 32, 32, kernel_size=3, stride=1, padding=(0, 0), bias=False)
            self.conv3 = binarized_modules_multi.BinarizeConv2d(bitwidth, bitwidth, 32, 32, kernel_size=3, stride=1, padding=(0, 0), bias=False)
            self.conv4 = binarized_modules_multi.BinarizeConv2d(bitwidth, bitwidth, 32, 64, kernel_size=2, stride=1, padding=(0, 0), bias=False)

            self.fc1 = binarized_modules_multi.BinarizeLinear(bitwidth, bitwidth, 64, 128, bias=False)
            self.fc2 = binarized_modules_multi.BinarizeLinear(bitwidth, bitwidth, 128, 2, bias=False)
            self.fc3 = binarized_modules_multi.BinarizeLinear(bitwidth, bitwidth, 128, 4, bias=False)
            self.fc4 = binarized_modules_multi.BinarizeLinear(bitwidth, bitwidth, 128, 10, bias=False)
            """
            self.fc2 = nn.Linear(128, 2, bias=False)
            self.fc3 = nn.Linear(128, 4, bias=False)
            self.fc4 = nn.Linear(128, 10, bias=False)
            """
            """
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=(0, 0), bias=False)
            self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=(0, 0), bias=False)
            self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=(0, 0), bias=False)
            self.conv4 = nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=(0, 0), bias=False)

            self.fc1 = nn.Linear(64, 128, bias=False)
            self.fc2 = nn.Linear(128, 2, bias=False)
            self.fc3 = nn.Linear(128, 4, bias=False)
            self.fc4 = nn.Linear(128, 10, bias=False)
            """
        else:
            if self.binarize_input:
              self.conv1 = binarized_modules_multi.TernarizeConv2d(conv_thres, input_bitwidth, bitwidth, 3, 32, kernel_size=3, stride=1, padding=(0, 0), align=False, bias=False)
            else:
              self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=(0, 0), align=False, bias=False)
            self.conv2 = binarized_modules_multi.TernarizeConv2d(conv_thres, bitwidth, bitwidth, 32, 32, kernel_size=3, stride=1, padding=(0, 0), align=self.align, bias=False)
            self.conv3 = binarized_modules_multi.TernarizeConv2d(conv_thres, bitwidth, bitwidth, 32, 32, kernel_size=3, stride=1, padding=(0, 0), align=self.align, bias=False)
            self.conv4 = binarized_modules_multi.TernarizeConv2d(conv_thres, bitwidth, bitwidth, 32, 64, kernel_size=2, stride=1, padding=(0, 0), align=self.align, bias=False)

            self.fc1 = binarized_modules_multi.TernarizeLinear(first_sparsity, bitwidth, bitwidth, 64, 128, align=self.align, bias=False)
            #self.fc2 = binarized_modules_multi.TernarizeLinear(rest_sparsity, bitwidth, bitwidth, 128, 2, align=self.align, bias=False)
            #self.fc3 = binarized_modules_multi.TernarizeLinear(rest_sparsity, bitwidth, bitwidth, 128, 4, align=self.align, bias=False)
            #self.fc4 = binarized_modules_multi.TernarizeLinear(rest_sparsity, bitwidth, bitwidth, 128, 10, align=self.align, bias=False)
            self.fc2 = nn.Linear(128, 2, bias=False)
            self.fc3 = nn.Linear(128, 4, bias=False)
            self.fc4 = nn.Linear(128, 10, bias=False)

        self.softmax1 = torch.nn.Softmax(dim = 1)

        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=3, padding=(0, 0))
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=3, padding=(0, 0))
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=(0, 0))

        self.act = F.relu if self.full else nn.Hardtanh()

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm1d(128)
        self.bn6 = nn.BatchNorm1d(2)
        self.bn7 = nn.BatchNorm1d(4)
        self.bn8 = nn.BatchNorm1d(10)
        
    def forward(self, x):
        x = torch.tensor(torch.full((1, 3, 48, 48), 3.))
        
        conv_layer = self.conv1
        bn_layer = self.bn1
        pool_layer = self.pool1

        print([param.shape for param in conv_layer.parameters()])

        x_ = x
        x_ = torch.transpose(x_, 1, 3)
        x_ = torch.transpose(x_, 1, 2)
        print("input (nwhc)", torch.flatten(x_)[:10])

        # converting nchw to nhwc (nchw -> nwhc -> nhwc)
        conv_parameters = conv_layer.weight
        conv_parameters = torch.transpose(conv_parameters, 1, 3)
        conv_parameters = torch.transpose(conv_parameters, 1, 2)
        parameters = self.fc2.weight
        #make parameters nchw for conv1
        print("parameter values (nhwc):", torch.flatten(parameters)[:27])
        with torch.no_grad():
          print("parameter count of nonnegative", str(int(torch.count_nonzero(torch.greater_equal(parameters, 0)))) + '/' + str(len(torch.flatten(parameters))))


        #x = bn_layer(conv_layer(x))
        x = self.pool1(self.bn1(self.conv1(x)))
        x = self.pool2(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.bn4(self.conv4(x))
        x = torch.reshape(x, (-1, 64))
        x = binarized_modules_multi.Binarize(x-0.01, quant_mode='det', bitwidth=1) # -0.01 to count 0 as negative
        #expected_out = torch.nn.functional.linear(x, parameters).flatten()
        #x.fill_(1)
        x = self.bn5(self.fc1(x))
        x = binarized_modules_multi.Binarize(x-0.01, quant_mode='det', bitwidth=1)
        print("input values (nhwc)", [elem.item() for elem in list(torch.flatten(x)[:64])])
        x1 = self.fc2(x)#self.bn6(self.fc2(x))
        x2 = self.bn7(self.fc3(x))
        x3 = self.bn8(self.fc4(x))
        x = x1
        #print("output values before binarization (nhwc)", [elem.item() for elem in list(torch.flatten(x)[:64])])
        #x = self.fc1(x)
        """
        x = conv_layer(x)
        x = bn_layer(x)
        if pool_layer:
          x = pool_layer(x)
        x = binarized_modules_multi.Binarize(x, quant_mode='det', bitwidth=1)
        """

        x_ = x
        """
        # converting nchw to nhwc
        x_ = torch.transpose(x_, 1, 3)
        x_ = torch.transpose(x_, 1, 2)
        """
        print("output values (nhwc)", [elem.item() for elem in list(torch.flatten(x_)[:128])])

        with torch.no_grad():
          print("output count of nonnegative", str(int(torch.count_nonzero(torch.greater_equal(x_, 0)))) + '/' + str(len(torch.flatten(x_))))
        # converting nchw to nhwc

        """
        x = self.act(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.act(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        x = torch.reshape(x, (-1, 64))
        x = self.act(self.bn5(self.fc1(x)))
        
        out1 = self.softmax1(self.bn6(self.fc2(x)))
        out2 = self.bn7(self.fc3(x))
        out3 = self.bn8(self.fc4(x))
        
        return out1, out2, out3
        """

"""
class VGGM(nn.Module):
    def __init__(self, n_classes=1251, full=False, binary=True, first_sparsity=0.8, rest_sparsity=0.9, conv_thres=0.7, align=True, bitwidth=1, input_bitwidth=1, binarize_input_True):
        super(VGGM, self).__init__()
        self.n_classes=n_classes
        if full:
            self.features=nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(in_channels=1, out_channels=96, kernel_size=(7,7), stride=(2,2), padding=1)),
                ('bn1', nn.BatchNorm2d(96, momentum=0.5)),
                ('relu1', nn.ReLU()),
                ('mpool1', nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))),
                ('conv2', nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5,5), stride=(2,2), padding=1)),
                ('bn2', nn.BatchNorm2d(256, momentum=0.5)),
                ('relu2', nn.ReLU()),
                ('mpool2', nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))),
                ('conv3', nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3,3), stride=(1,1), padding=1)),
                ('bn3', nn.BatchNorm2d(384, momentum=0.5)),
                ('relu3', nn.ReLU()),
                ('conv4', nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=1)),
                ('bn4', nn.BatchNorm2d(256, momentum=0.5)),
                ('relu4', nn.ReLU()),
                ('conv5', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=1)),
                ('bn5', nn.BatchNorm2d(256, momentum=0.5)),
                ('relu5', nn.ReLU()),
                ('mpool5', nn.MaxPool2d(kernel_size=(5,3), stride=(3,2))),
                ('fc6', nn.Conv2d(in_channels=256, out_channels=4096, kernel_size=(9,1), stride=(1,1))),
                ('bn6', nn.BatchNorm2d(4096, momentum=0.5)),
                ('relu6', nn.ReLU()),
                ('apool6', nn.AdaptiveAvgPool2d((1,1))),
                ('flatten', nn.Flatten())]))
                
            self.classifier=nn.Sequential(OrderedDict([
                ('fc7', nn.Linear(4096, 1024)),
                #('drop1', nn.Dropout()),
                ('relu7', nn.ReLU()),
                ('fc8', nn.Linear(1024, n_classes))]))
        elif binary:
            self.features=nn.Sequential(OrderedDict([
                ('conv1', binarized_modules_multi.BinarizeConv2d(input_bitwidth, bitwidth, in_channels=1, out_channels=96, kernel_size=(7,7), stride=(2,2), padding=1)),
                ('bn1', nn.BatchNorm2d(96, momentum=0.5)),
                ('relu1', nn.functional.tanh),
                ('mpool1', nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))),
                ('conv2', binarized_modules_multi.BinarizeConv2d(bitwidth, bitwidth, in_channels=96, out_channels=256, kernel_size=(5,5), stride=(2,2), padding=1)),
                ('bn2', nn.BatchNorm2d(256, momentum=0.5)),
                ('relu2', nn.functional.tanh),
                ('mpool2', nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))),
                ('conv3', binarized_modules_multi.BinarizeConv2d(bitwidth, bitwidth, in_channels=256, out_channels=384, kernel_size=(3,3), stride=(1,1), padding=1)),
                ('bn3', nn.BatchNorm2d(384, momentum=0.5)),
                ('relu3', nn.functional.tanh),
                ('conv4', binarized_modules_multi.BinarizeConv2d(bitwidth, bitwidth, in_channels=384, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=1)),
                ('bn4', nn.BatchNorm2d(256, momentum=0.5)),
                ('relu4', nn.functional.tanh),
                ('conv5', binarized_modules_multi.BinarizeConv2d(bitwidth, bitwidth, in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=1)),
                ('bn5', nn.BatchNorm2d(256, momentum=0.5)),
                ('relu5', nn.functional.tanh),
                ('mpool5', nn.MaxPool2d(kernel_size=(5,3), stride=(3,2))),
                ('fc6', binarized_modules_multi.BinarizeConv2d(bitwidth, bitwidth, in_channels=256, out_channels=4096, kernel_size=(9,1), stride=(1,1))),
                ('bn6', nn.BatchNorm2d(4096, momentum=0.5)),
                ('relu6', nn.functional.tanh),
                ('apool6', nn.AdaptiveAvgPool2d((1,1))),
                ('flatten', nn.Flatten())]))
            self.classifier=nn.Sequential(OrderedDict([
                ('fc7', binarized_modules_multi.BinarizeLinear(bitwidth, bitwidth, 4096, 1024, bias=False)),
                #('drop1', nn.Dropout()),
                ('relu7', nn.functional.tanh),
                ('fc8', binarized_modules_multi.BinarizeLinear(bitwidth, bitwidth, 1024, n_classes, bias=False))]))
        else:
            #TODO
            pass        

    def forward(self, inp):
        inp=self.features(inp)
        #inp=inp.view(inp.size()[0],-1)
        inp=self.classifier(inp)
        return inp
"""

class FC_small(nn.Module):
    def __init__(self, full=False, binary=True, first_sparsity=0.8, rest_sparsity=0.9, hid=512, ind=784, align=False):
        super(FC_small, self).__init__()
        self.align = align
        self.pruned = False
        self.hid = hid
        self.ind = ind
        
        full = True
        self.full = full
        self.binary = binary
        
        if full:
            self.fc1 = nn.Linear(ind, hid)
            self.fc2 = nn.Linear(hid, 10)
        elif binary:
            self.fc1 = binarized_modules.BinarizeLinear(ind, hid)
            self.fc2 = binarized_modules.BinarizeLinear(hid, 10)
        else:
            self.fc1 = binarized_modules.TernarizeLinear(first_sparsity, ind, hid, align=align)
            self.fc2 = binarized_modules.TernarizeLinear(rest_sparsity, hid, 10, align=align)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(hid)
            
        self.bn2 = nn.BatchNorm1d(10, affine=True)
        self.logsoftmax=nn.LogSoftmax(dim=1)
    def forward(self, x):
        if self.full:
            x = x.view(-1, 784)
            if x.size(1)==784:
                x = x[:,:768]
            x = F.relu(self.fc1(x))
            x = self.bn1(x)
            x = self.fc2(x)
        else:
            x = x.view(-1, 784)
            if x.size(1)==784:
                x = x[:,:768]
            if self.binary:
                x = self.fc1(x)
            else:
                x = self.fc1(x, self.pruned)
            x = self.bn1(x)
            self.htanh1(x)
            if self.binary:
                x = self.fc2(x)
            else:
                x = self.fc2(x, self.pruned)
            x = self.bn2(x)
        return x
    
class FC_large(nn.Module):
    def __init__(self, full=False, binary=True, first_sparsity=0.8, rest_sparsity=0.9, hid=4096, ind=768, align=False):
        super(FC_large, self).__init__()
        self.align = align
        self.pruned = False
        self.hid = hid
        self.ind = ind
        
        self.full = full
        self.binary = binary
        
        if full:
            self.fc1 = nn.Linear(ind, hid)
            self.fc2 = nn.Linear(hid, hid)
            self.fc3 = nn.Linear(hid, hid)
            self.fc4 = nn.Linear(hid, 10)
        elif binary:
            self.fc1 = binarized_modules.BinarizeLinear(ind, hid)
            self.fc2 = binarized_modules.BinarizeLinear(hid, hid)
            self.fc3 = binarized_modules.BinarizeLinear(hid, hid)
            self.fc4 = binarized_modules.BinarizeLinear(hid, 10)
        else:
            self.fc1 = binarized_modules.TernarizeLinear(first_sparsity, ind, hid, align=align)
            self.fc2 = binarized_modules.TernarizeLinear(rest_sparsity, hid, hid, align=align)
            self.fc3 = binarized_modules.TernarizeLinear(rest_sparsity, hid, hid, align=align)
            self.fc4 = binarized_modules.TernarizeLinear(rest_sparsity, hid, 10, align=align)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(hid)
            
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(hid)
            
        self.htanh3 = nn.Hardtanh()
        self.bn3 = nn.BatchNorm1d(hid)
            
        self.bn4 = nn.BatchNorm1d(10, affine=True)
        self.logsoftmax=nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        if self.full:
            x = x.view(-1, 784)
            if x.size(1)==784:
                x = x[:,:768]
            x = F.relu(self.fc1(x))
            x = self.bn1(x)
            x = F.relu(self.fc2(x))
            x = self.bn2(x)
            x = F.relu(self.fc3(x))
            x = self.bn3(x)
            x = self.fc4(x)
        else:
            x = x.view(-1, 784)
            if x.size(1)==784:
                x = x[:,:768]
            if self.binary:
                x = self.fc1(x)
            else:
                x = self.fc1(x, self.pruned)
            x = self.bn1(x)
            self.htanh1(x)
            if self.binary:
                x = self.fc2(x)
            else:
                x = self.fc2(x, self.pruned)
            x = self.bn2(x)
            self.htanh2(x)
            if self.binary:
                x = self.fc3(x)
            else:
                x = self.fc3(x, self.pruned)
            x = self.bn3(x)
            self.htanh3(x)
            if self.binary:
                x = self.fc4(x)
            else:
                x = self.fc4(x, self.pruned)
            x = self.bn4(x)
        return self.logsoftmax(x)
    
class CNN_medium(nn.Module):

    def __init__(self, full=False, binary=True, conv_thres=0.7, fc_thres=0.9, align=False, pad=0):
        super(CNN_medium, self).__init__()
        
        self.pruned = False
        self.full = full
        self.binary = binary
        self.pad = pad
        
        self.conv1 = binarized_modules.BinarizeConv2d(3, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.htanh1 = nn.Hardtanh(inplace=True)
        
        if full:
            self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
            self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
            self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
            self.fc1 = nn.Linear(512*4*4, 10)
        elif binary:
            self.conv1 = binarized_modules.BinarizeConv2d(3, 128, kernel_size=3, stride=1, padding=0)
            self.conv2 = binarized_modules.BinarizeConv2d(128, 128, kernel_size=3, stride=1, padding=0)
            self.conv3 = binarized_modules.BinarizeConv2d(128, 256, kernel_size=3, stride=1, padding=0)
            self.conv4 = binarized_modules.BinarizeConv2d(256, 256, kernel_size=3, stride=1, padding=0)
            self.conv5 = binarized_modules.BinarizeConv2d(256, 512, kernel_size=3, stride=1, padding=0)
            self.conv6 = binarized_modules.BinarizeConv2d(512, 512, kernel_size=3, stride=1, padding=0)
            self.conv7 = binarized_modules.BinarizeConv2d(512, 10, kernel_size=4, padding=0)
            self.fc1 = binarized_modules.BinarizeLinear(1024, 1024)
        else:
            self.conv1 = binarized_modules.BinarizeConv2d(3, 128, kernel_size=3, stride=1, padding=0)
            self.conv2 = binarized_modules.TernarizeConv2d(conv_thres, 128, 128, kernel_size=3, padding=0, align=align)
            self.conv3 = binarized_modules.TernarizeConv2d(conv_thres, 128, 256, kernel_size=3, padding=0, align=align)
            self.conv4 = binarized_modules.TernarizeConv2d(conv_thres, 256, 256, kernel_size=3, padding=0, align=align)
            self.conv5 = binarized_modules.TernarizeConv2d(conv_thres, 256, 512, kernel_size=3, padding=0, align=align)
            self.conv6 = binarized_modules.TernarizeConv2d(conv_thres, 512, 512, kernel_size=3, padding=0, align=align)
            self.conv7 = binarized_modules.TernarizeConv2d(0.49, 512, 10, kernel_size=4, padding=0, align=align)
            self.fc1 = binarized_modules.BinarizeLinear(1024, 1024)
            
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.htanh2 = nn.Hardtanh(inplace=True)
        
        
        self.bn3 = nn.BatchNorm2d(256)
        self.htanh3 = nn.Hardtanh(inplace=True)
        
        
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn4 = nn.BatchNorm2d(256)
        self.htanh4 = nn.Hardtanh(inplace=True)
        
        
        self.bn5 = nn.BatchNorm2d(512)
        self.htanh5 = nn.Hardtanh(inplace=True)
        
        
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn6 = nn.BatchNorm2d(512)
        self.htanh6 = nn.Hardtanh(inplace=True)
        
        
        self.bnfc1 = nn.BatchNorm1d(10, affine=True)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        self.regime = {
            0: {'optimizer': 'Adam', 'betas': (0.9, 0.999),'lr': 5e-3},
            40: {'lr': 1e-3},
            80: {'lr': 5e-4},
            100: {'lr': 1e-4},
            120: {'lr': 5e-5},
            140: {'lr': 1e-5}
        }

    def forward(self, x):
        if self.full:
            x = F.relu(self.conv1(x))
            x = self.bn1(x)
            
            x = F.relu(self.conv2(x))
            x = self.pool2(x)
            x = self.bn2(x)
            
            x = F.relu(self.conv3(x))
            x = self.bn3(x)
            
            x = F.relu(self.conv4(x))
            x = self.pool4(x)
            x = self.bn4(x)
            
            x = F.relu(self.conv5(x))
            x = self.bn5(x)
            
            x = F.relu(self.conv6(x))
            x = self.pool6(x)
            x = self.bn6(x)
            
            x = x.view(-1, 512*4*4)
            x = F.relu(self.fc1(x))
            self.fc1_result = x.data.clone()
        else:
            x = F.pad(x, (1,1,1,1), value=self.pad)
            x = self.conv1(x)
            x = self.bn1(x)
            self.htanh1(x)
            
            x = F.pad(x, (1,1,1,1), value=self.pad)
            if self.binary:
                x = self.conv2(x)
            else:
                x = self.conv2(x, self.pruned)
            x = self.pool2(x)
            x = self.bn2(x)
            self.htanh2(x)
            
            x = F.pad(x, (1,1,1,1), value=self.pad)
            if self.binary:
                x = self.conv3(x)
            else:
                x = self.conv3(x, self.pruned)
            x = self.bn3(x)
            self.htanh3(x)
            
            x = F.pad(x, (1,1,1,1), value=self.pad)
            if self.binary:
                x = self.conv4(x)
            else:
                x = self.conv4(x, self.pruned)
            x = self.pool4(x)
            x = self.bn4(x)
            self.htanh4(x)
            
            x = F.pad(x, (1,1,1,1), value=self.pad)
            if self.binary:
                x = self.conv5(x)
            else:
                x = self.conv5(x, self.pruned)
            x = self.bn5(x)
            self.htanh5(x)
            
            x = F.pad(x, (1,1,1,1), value=self.pad)
            if self.binary:
                x = self.conv6(x)
            else:
                x = self.conv6(x, self.pruned)
            x = self.pool6(x)
            x = self.bn6(x)
            self.htanh6(x)
            
            if self.binary:
                x = self.conv7(x)
            else:
                x = self.conv7(x, self.pruned)
            x = x.view(-1, 10)
            x = self.bnfc1(x)
        return self.logsoftmax(x)
    
class CNN_large(nn.Module):

    def __init__(self, full=False, binary=True, conv_thres=0.7, fc_thres=0.9, align=False, pad=0):
        super(CNN_large, self).__init__()
        
        self.pruned = False
        self.full = full
        self.binary = binary
        self.pad = pad
        
        self.conv1 = binarized_modules.BinarizeConv2d(3, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.htanh1 = nn.Hardtanh(inplace=True)
        
        if full:
            self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
            self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
            self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
            self.fc1 = nn.Linear(512*4*4, 1024)
            self.fc2 = nn.Linear(1024, 1024)
            self.fc3 = nn.Linear(1024, 10)
        elif binary:
            self.conv1 = binarized_modules.BinarizeConv2d(3, 128, kernel_size=3, stride=1, padding=0)
            self.conv2 = binarized_modules.BinarizeConv2d(128, 128, kernel_size=3, stride=1, padding=0)
            self.conv3 = binarized_modules.BinarizeConv2d(128, 256, kernel_size=3, stride=1, padding=0)
            self.conv4 = binarized_modules.BinarizeConv2d(256, 256, kernel_size=3, stride=1, padding=0)
            self.conv5 = binarized_modules.BinarizeConv2d(256, 512, kernel_size=3, stride=1, padding=0)
            self.conv6 = binarized_modules.BinarizeConv2d(512, 512, kernel_size=3, stride=1, padding=0)
            self.conv7 = binarized_modules.BinarizeConv2d(512, 1024, kernel_size=4, padding=0)
            self.fc1 = binarized_modules.BinarizeLinear(1024, 1024)
            self.fc2 = binarized_modules.BinarizeLinear(1024, 10)
        else:
            self.conv1 = binarized_modules.BinarizeConv2d(3, 128, kernel_size=3, stride=1, padding=0)
            self.conv2 = binarized_modules.TernarizeConv2d(conv_thres, 128, 128, kernel_size=3, padding=0, align=align)
            self.conv3 = binarized_modules.TernarizeConv2d(conv_thres, 128, 256, kernel_size=3, padding=0, align=align)
            self.conv4 = binarized_modules.TernarizeConv2d(conv_thres, 256, 256, kernel_size=3, padding=0, align=align)
            self.conv5 = binarized_modules.TernarizeConv2d(conv_thres, 256, 512, kernel_size=3, padding=0, align=align)
            self.conv6 = binarized_modules.TernarizeConv2d(conv_thres, 512, 512, kernel_size=3, padding=0, align=align)
            self.conv7 = binarized_modules.TernarizeConv2d(fc_thres, 512, 1024, kernel_size=4, padding=0, align=align)
            self.fc1 = binarized_modules.TernarizeLinear(fc_thres, 1024, 1024, align=align)
            self.fc2 = binarized_modules.TernarizeLinear(0.49, 1024, 10)
            
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.htanh2 = nn.Hardtanh(inplace=True)
        
        
        self.bn3 = nn.BatchNorm2d(256)
        self.htanh3 = nn.Hardtanh(inplace=True)
        
        
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn4 = nn.BatchNorm2d(256)
        self.htanh4 = nn.Hardtanh(inplace=True)
        
        
        self.bn5 = nn.BatchNorm2d(512)
        self.htanh5 = nn.Hardtanh(inplace=True)
        
        
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn6 = nn.BatchNorm2d(512)
        self.htanh6 = nn.Hardtanh(inplace=True)
        
        
        self.bnfc1 = nn.BatchNorm1d(1024)
        self.htanhfc1 = nn.Hardtanh(inplace=True)
        
        
        self.bnfc2 = nn.BatchNorm1d(1024)
        self.htanhfc2 = nn.Hardtanh(inplace=True)
        
        
        self.bnfc3 = nn.BatchNorm1d(10, affine=True)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        self.regime = {
            0: {'optimizer': 'Adam', 'betas': (0.9, 0.999),'lr': 5e-3},
            40: {'lr': 1e-3},
            80: {'lr': 5e-4},
            100: {'lr': 1e-4},
            120: {'lr': 5e-5},
            140: {'lr': 1e-5}
        }

    def forward(self, x):
        if self.full:
            x = F.relu(self.conv1(x))
            x = self.bn1(x)
            
            x = F.relu(self.conv2(x))
            x = self.pool2(x)
            x = self.bn2(x)
            
            x = F.relu(self.conv3(x))
            x = self.bn3(x)
            
            x = F.relu(self.conv4(x))
            x = self.pool4(x)
            x = self.bn4(x)
            
            x = F.relu(self.conv5(x))
            x = self.bn5(x)
            
            x = F.relu(self.conv6(x))
            x = self.pool6(x)
            x = self.bn6(x)
            
            x = x.view(-1, 512*4*4)
            x = F.relu(self.fc1(x))
            x = self.bnfc1(x)
            
            x = F.relu(self.fc2(x))
            x = self.bnfc2(x)
            
            x = F.relu(self.fc3(x))
            self.fc3_result = x.data.clone()
        else:
            x = F.pad(x, (1,1,1,1), value=self.pad)
            x = self.conv1(x)
            x = self.bn1(x)
            self.htanh1(x)
            
            x = F.pad(x, (1,1,1,1), value=self.pad)
            if self.binary:
                x = self.conv2(x)
            else:
                x = self.conv2(x, self.pruned)
            x = self.pool2(x)
            x = self.bn2(x)
            self.htanh2(x)
            
            x = F.pad(x, (1,1,1,1), value=self.pad)
            if self.binary:
                x = self.conv3(x)
            else:
                x = self.conv3(x, self.pruned)
            x = self.bn3(x)
            self.htanh3(x)
            
            x = F.pad(x, (1,1,1,1), value=self.pad)
            if self.binary:
                x = self.conv4(x)
            else:
                x = self.conv4(x, self.pruned)
            x = self.pool4(x)
            x = self.bn4(x)
            self.htanh4(x)
            
            x = F.pad(x, (1,1,1,1), value=self.pad)
            if self.binary:
                x = self.conv5(x)
            else:
                x = self.conv5(x, self.pruned)
            x = self.bn5(x)
            self.htanh5(x)
            
            x = F.pad(x, (1,1,1,1), value=self.pad)
            if self.binary:
                x = self.conv6(x)
            else:
                x = self.conv6(x, self.pruned)
            x = self.pool6(x)
            x = self.bn6(x)
            self.htanh6(x)
            
            if self.binary:
                x = self.conv7(x)
            else:
                x = self.conv7(x, self.pruned)
            print(x.shape)
            x = x.view(-1, 1024)
            x = self.bnfc1(x)
            self.htanhfc1(x)

            if self.binary:
                x = self.fc1(x)
            else:
                x = self.fc1(x, self.pruned)
            x = self.bnfc2(x)
            self.htanhfc2(x)
            
            if self.binary:
                x = self.fc2(x)
            else:
                x = self.fc2(x, self.pruned)
            x = self.bnfc3(x)
        return self.logsoftmax(x)
    
class CNN_tiny(nn.Module):

    def __init__(self, full=False, binary=True, conv_thres=0.7, fc_thres=0.9, align=False):
        super(CNN_tiny, self).__init__()
        
        self.pruned = False
        self.full = full
        self.binary = binary
        
        if full:
            self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=0)
            self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=0)
            self.fc1 = nn.Linear(32*4*4, 10)
        elif binary:
            self.conv1 = binarized_modules.BinarizeConv2d(1, 32, kernel_size=5, stride=1, padding=0)
            self.conv2 = binarized_modules.BinarizeConv2d(32, 32, kernel_size=5, stride=1, padding=0)
            self.fc1 = binarized_modules.BinarizeConv2d(32, 10, kernel_size=4, padding=0)
        else:
            self.conv1 = binarized_modules.BinarizeConv2d(1, 32, kernel_size=5, stride=1, padding=0)
            self.conv2 = binarized_modules.TernarizeConv2d(conv_thres, 32, 32, kernel_size=5, stride=1, padding=0, align=align)
            self.fc1 = binarized_modules.TernarizeConv2d(0.49, 32, 10, kernel_size=4, padding=0, align=align)
            
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.htanh1 = nn.Hardtanh(inplace=True)
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.htanh2 = nn.Hardtanh(inplace=True)
     
        self.bnfc1 = nn.BatchNorm1d(10, affine=True)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        if self.full:
            x = F.relu(self.conv1(x))
            x = self.pool1(x)
            x = self.bn1(x)
            
            x = F.relu(self.conv2(x))
            x = self.pool2(x)
            x = self.bn2(x)
            
            x = x.view(-1, 32*4*4)
            x = F.relu(self.fc1(x))
        else:
            x = self.conv1(x)
            x = self.pool1(x)
            x = self.bn1(x)
            self.htanh1(x)
            
            if self.binary:
                x = self.conv2(x)
            else:
                x = self.conv2(x, self.pruned)
            x = self.pool2(x)
            x = self.bn2(x)
            self.htanh2(x)
            
            if self.binary:
                x = self.fc1(x)
            else:
                x = self.fc1(x, self.pruned)
            x = x.view(-1, 10)
            x = self.bnfc1(x)
        return self.logsoftmax(x)
    
#Deep autoencoder model
#https://github.com/mlcommons/tiny/tree/master/v0.5
class DeepAutoEncoder(nn.Module):
    def __init__(self, full=True, binary=True, sparsity=0.1, align=False):
      super(DeepAutoEncoder, self).__init__()
      
      self.full = full
      self.binary = binary
      self.align = align
      self.pruned = False

      ind = 640
      hid = 128
      mid = 8

      self.act = F.relu if self.full else nn.Hardtanh()
      
      if full:
        self.fc1 = nn.Linear(ind, hid, bias=False)
        self.fc2 = nn.Linear(hid, hid, bias=False)
        self.fc3 = nn.Linear(hid, hid, bias=False)
        self.fc4 = nn.Linear(hid, hid, bias=False)
        self.fc5 = nn.Linear(hid, mid, bias=False)
        self.fc6 = nn.Linear(mid, hid, bias=False)
        self.fc7 = nn.Linear(hid, hid, bias=False)
        self.fc8 = nn.Linear(hid, hid, bias=False)
        self.fc9 = nn.Linear(hid, hid, bias=False)
        self.fc10 = nn.Linear(hid, ind, bias=False)
      elif binary:
        self.fc1 = nn.Linear(ind, hid, bias=False)
        self.fc2 = binarized_modules_multi.BinarizeLinear(1, 1, hid, hid, bias=False)
        self.fc3 = binarized_modules_multi.BinarizeLinear(1, 1, hid, hid, bias=False)
        self.fc4 = binarized_modules_multi.BinarizeLinear(1, 1, hid, hid, bias=False)
        self.fc5 = binarized_modules_multi.BinarizeLinear(1, 1, hid, mid, bias=False)
        self.fc6 = binarized_modules_multi.BinarizeLinear(1, 1, mid, hid, bias=False)
        self.fc7 = binarized_modules_multi.BinarizeLinear(1, 1, hid, hid, bias=False)
        self.fc8 = binarized_modules_multi.BinarizeLinear(1, 1, hid, hid, bias=False)
        self.fc9 = binarized_modules_multi.BinarizeLinear(1, 1, hid, hid, bias=False)
        self.fc10 = nn.Linear(hid, ind, bias=False)
      else:
        self.fc1 = binarized_modules_multi.TernarizeLinear(sparsity, 1, 1, ind, hid, bias=False, align=self.align)
        self.fc2 = binarized_modules_multi.TernarizeLinear(sparsity, 1, 1, hid, hid, bias=False, align=self.align)
        self.fc3 = binarized_modules_multi.TernarizeLinear(sparsity, 1, 1, hid, hid, bias=False, align=self.align)
        self.fc4 = binarized_modules_multi.TernarizeLinear(sparsity, 1, 1, hid, hid, bias=False, align=self.align)
        self.fc5 = binarized_modules_multi.TernarizeLinear(sparsity, 1, 1, hid, mid, bias=False, align=self.align)
        self.fc6 = binarized_modules_multi.TernarizeLinear(sparsity, 1, 1, mid, hid, bias=False, align=self.align)
        self.fc7 = binarized_modules_multi.TernarizeLinear(sparsity, 1, 1, hid, hid, bias=False, align=self.align)
        self.fc8 = binarized_modules_multi.TernarizeLinear(sparsity, 1, 1, hid, hid, bias=False, align=self.align)
        self.fc9 = binarized_modules_multi.TernarizeLinear(sparsity, 1, 1, hid, hid, bias=False, align=self.align)
        self.fc10 = binarized_modules_multi.TernarizeLinear(sparsity, 1, 1, hid, ind, bias=False, align=self.align)

      self.batchnorm1 = nn.BatchNorm1d(hid)
      self.batchnorm2 = nn.BatchNorm1d(hid)
      self.batchnorm3 = nn.BatchNorm1d(hid)
      self.batchnorm4 = nn.BatchNorm1d(hid)
      self.batchnorm5 = nn.BatchNorm1d(mid)
      self.batchnorm6 = nn.BatchNorm1d(hid)
      self.batchnorm7 = nn.BatchNorm1d(hid)
      self.batchnorm8 = nn.BatchNorm1d(hid)
      self.batchnorm9 = nn.BatchNorm1d(hid)
      self.batchnorm10 = nn.BatchNorm1d(ind)
    def forward(self, x):
        x = self.fc1(x)
        x = self.batchnorm1(x)
        x = self.act(x)

        x = self.fc2(x)
        x = self.batchnorm2(x)
        x = self.act(x)
        
        x = self.fc3(x)
        x = self.batchnorm3(x)
        x = self.act(x)

        x = self.fc4(x)
        x = self.batchnorm4(x)
        x = self.act(x)

        x = self.fc5(x)
        x = self.batchnorm5(x)
        x = self.act(x)

        x = self.fc6(x)
        x = self.batchnorm6(x)
        x = self.act(x)
        
        x = self.fc7(x)
        x = self.batchnorm7(x)
        x = self.act(x)

        x = self.fc8(x)
        x = self.batchnorm8(x)
        x = self.act(x)

        x = self.fc9(x)
        x = self.batchnorm9(x)
        x = self.act(x)

        x = self.fc10(x)
        x = self.batchnorm10(x)
        #x = self.act(x)
        
        return x 
