import torch

import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import utils
import utils_own

import binarized_modules

class pnet(nn.Module):
    def __init__(self, full=False, binary=True, conv_thres=0.7, align=False):
        super(pnet, self).__init__()
        self.align = align
        self.pruned = False

        self.full = full
        self.binary = binary

        """
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3, stride=1, padding='valid')
        self.conv2 = nn.Conv2d(10, 16, kernel_size=3, stride=1, padding='valid')
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding='valid')
        self.conv4 = nn.Conv2d(32, 2, kernel_size=1, stride=1, padding='valid')
        self.conv5 = nn.Conv2d(32, 4, kernel_size=1, stride=1, padding='valid')
        """

        if full:
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding='valid')
            self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding='valid')
            self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding='valid')
            self.conv4 = nn.Conv2d(32, 2, kernel_size=1, stride=1, padding='valid')
            self.conv5 = nn.Conv2d(32, 4, kernel_size=1, stride=1, padding='valid')
        elif binary:
            self.conv1 = binarized_modules.BinarizeConv2d(3, 32, kernel_size=3, stride=1, padding='valid')
            self.conv2 = binarized_modules.BinarizeConv2d(32, 32, kernel_size=3, stride=1, padding='valid')
            self.conv3 = binarized_modules.BinarizeConv2d(32, 32, kernel_size=3, stride=1, padding='valid')
            self.conv4 = binarized_modules.BinarizeConv2d(32, 2, kernel_size=1, stride=1, padding='valid')
            self.conv5 = nn.Conv2d(32, 4, kernel_size=1, stride=1, padding='valid')
        """
        else:
            self.conv1 = binarized_modules.BinarizeConv2d(3, 32, kernel_size=3, stride=1, padding='valid')
            self.conv2 = binarized_modules.TernarizeConv2d(conv_thres, 32, 32, kernel_size=3, stride=1, padding='valid', align=align)
            self.conv3 = binarized_modules.TernarizeConv2d(conv_thres, 32, 32, kernel_size=3, stride=1, padding='valid', align=align)
            self.conv4 = binarized_modules.BinarizeConv2d(32, 2, kernel_size=1, stride=1, padding='valid')
            self.conv5 = binarized_modules.BinarizeConv2d(32, 4, kernel_size=1, stride=1, padding='valid')
        """


        self.softmax1 = torch.nn.Softmax(dim = 1)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.htanh1 = nn.Hardtanh(inplace=True)
        self.htanh2 = nn.Hardtanh(inplace=True)
        self.htanh3 = nn.Hardtanh(inplace=True)
        self.htanh4 = nn.Hardtanh(inplace=True)
        self.htanh5 = nn.Hardtanh(inplace=True)
        
        self.exc_count1 = 0
        self.exc_count2 = 0
        self.exc_count3 = 0
        self.total_count = 0
        
    def forward(self, x):
        #TODO: take full, binary parameters into account
        if self.full:
            x = F.relu(self.conv1(x))
            x = self.pool1(x)
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))

            out1 = self.softmax1(self.conv4(x))
            out2 = self.conv5(x)

            out1 = torch.reshape(out1, (-1, 2))
            out2 = torch.reshape(out2, (-1, 4))
        else:
            """
            index = 0
            if(torch.any(torch.tensor([torch.all(x[index]==0) for index in range(len(x))]))):
                #print("Label 1", self.exc_count, torch.tensor([torch.all(x[index]==0) for index in range(len(x))]))
                self.exc_count1 += int(torch.sum(torch.tensor([torch.all(x[index]==0) for index in range(len(x))])))
                #raise Exception("Stop 1")
            self.total_count += len(x)
            """

            #x = F.relu(self.conv1(x))
            x = self.conv1(x)
            x = self.pool1(x)
            self.htanh1(x)
            x = self.conv2(x)
            self.htanh2(x)
            x = self.conv3(x)
            self.htanh3(x)
            out1 = self.conv4(x)
            #self.htanh4(out1)
            out1 = self.softmax1(out1)
            out2 = self.conv5(x)
            self.htanh5(out2)

            out1 = torch.reshape(out1, (-1, 2))
            out2 = torch.reshape(out2, (-1, 4))

        #out1, out2 = out1.type(torch.DoubleTensor), out2.type(torch.DoubleTensor)
        return out1, out2    

class rnet(nn.Module):
    def __init__(self, full=False, binary=True, first_sparsity=0.8, rest_sparsity=0.9, conv_thres=0.7, align=False):
        super(rnet, self).__init__()
        self.align = align
        self.pruned = False

        self.full = full
        self.binary = binary

        if full:
            self.conv1 = nn.Conv2d(3, 28, kernel_size=3, stride=1, padding='valid')
            self.conv2 = nn.Conv2d(28, 48, kernel_size=3, stride=1, padding='valid')
            self.conv3 = nn.Conv2d(48, 64, kernel_size=2, stride=1, padding='valid')

            self.fc1 = nn.Linear(256, 128)
            self.fc2 = nn.Linear(128, 2)
            self.fc3 = nn.Linear(128, 4)
        elif binary:
            self.conv1 = binarized_modules.BinarizeConv2d(3, 28, kernel_size=3, stride=1, padding='valid')
            self.conv2 = binarized_modules.BinarizeConv2d(28, 48, kernel_size=3, stride=1, padding='valid')
            self.conv3 = binarized_modules.BinarizeConv2d(48, 64, kernel_size=2, stride=1, padding='valid')

            self.fc1 = binarized_modules.BinarizeLinear(256, 128)
            self.fc2 = binarized_modules.BinarizeLinear(128, 2)
            self.fc3 = nn.Linear(128, 4)
        """
        else:
            self.conv1 = binarized_modules.BinarizeConv2d(3, 28, kernel_size=3, stride=1, padding='valid')
            self.conv2 = binarized_modules.TernarizeConv2d(conv_thres, 28, 48, kernel_size=3, stride=1, padding='valid', align=align)
            self.conv3 = binarized_modules.TernarizeConv2d(conv_thres, 48, 64, kernel_size=2, stride=1, padding='valid', align=align)
            self.fc1 = binarized_modules.TernarizeLinear(first_sparsity, 256, 128, align=align)
            self.fc2 = binarized_modules.TernarizeLinear(rest_sparsity, 128, 2, align=align)
            self.fc3 = binarized_modules.TernarizeLinear(rest_sparsity, 128, 4, align=align)
        """

        self.softmax1 = torch.nn.Softmax(dim = 1)

        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=(0, 0)) #TODO: Find the padding for "SAME"
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=(0, 0))

        self.htanh1 = nn.Hardtanh(inplace=True)
        self.htanh2 = nn.Hardtanh(inplace=True)
        self.htanh3 = nn.Hardtanh(inplace=True)
        self.htanh4 = nn.Hardtanh(inplace=True)
        self.htanh5 = nn.Hardtanh(inplace=True)
        self.htanh6 = nn.Hardtanh(inplace=True)
        
    def forward(self, x):
        if self.full:
            x = F.relu(self.conv1(x))
            x = self.pool1(x)
            x = F.relu(self.conv2(x))
            x = self.pool2(x)
            x = F.relu(self.conv3(x))
            x = torch.flatten(x, start_dim=1)

            x = F.relu(self.fc1(x))
            
            out1 = self.softmax1(self.fc2(x))
            out2 = self.fc3(x)
        else:
            """
            if(torch.any(torch.tensor([torch.all(x[index]==0) for index in range(len(x))]))):
                #print("Label 1", self.exc_count, torch.tensor([torch.all(x[index]==0) for index in range(len(x))]))
                self.exc_count1 += int(torch.sum(torch.tensor([torch.all(x[index]==0) for index in range(len(x))])))
                #raise Exception("Stop 1")
            self.total_count += len(x)
            """

            x = self.conv1(x)
            x = self.pool1(x)
            self.htanh1(x)
            x = self.conv2(x)
            x = self.pool2(x)
            self.htanh2(x)
            x = self.conv3(x)
            self.htanh3(x)
            x = torch.flatten(x, start_dim=1)

            x = self.fc1(x)
            self.htanh4(x)

            out1 = self.fc2(x)
            #self.htanh5(x)
            out1 = self.softmax1(out1)
            
            out2 = self.fc3(x)
            self.htanh6(out2)

        return out1, out2

class onet(nn.Module):
    def __init__(self, full=False, binary=True, first_sparsity=0.8, rest_sparsity=0.9, conv_thres=0.7, align=False):
        super(onet, self).__init__()
        self.align = align
        self.pruned = False

        self.full = full
        self.binary = binary

        if full:
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding='valid')
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='valid')
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='valid')
            self.conv4 = nn.Conv2d(64, 128, kernel_size=2, stride=1, padding='valid')

            #TODO: bias should be true, not false
            self.fc1 = nn.Linear(1152, 256)
            self.fc2 = nn.Linear(256, 2)
            self.fc3 = nn.Linear(256, 4)
            self.fc4 = nn.Linear(256, 10)
        elif binary:
            self.conv1 = binarized_modules.BinarizeConv2d(3, 32, kernel_size=3, stride=1, padding='valid')
            self.conv2 = binarized_modules.BinarizeConv2d(32, 64, kernel_size=3, stride=1, padding='valid')
            self.conv3 = binarized_modules.BinarizeConv2d(64, 64, kernel_size=3, stride=1, padding='valid')
            self.conv4 = binarized_modules.BinarizeConv2d(64, 128, kernel_size=2, stride=1, padding='valid')

            self.fc1 = binarized_modules.BinarizeLinear(1152, 256)
            self.fc2 = binarized_modules.BinarizeLinear(256, 2)
            self.fc3 = nn.Linear(256, 4)
            self.fc4 = nn.Linear(256, 10)
        """
        else:
            self.conv1 = binarized_modules.BinarizeConv2d(3, 32, kernel_size=3, stride=1, padding='valid')
            self.conv2 = binarized_modules.TernarizeConv2d(conv_thres, 32, 64, kernel_size=3, stride=1, padding='valid', align=align)
            self.conv3 = binarized_modules.TernarizeConv2d(conv_thres, 64, 64, kernel_size=3, stride=1, padding='valid', align=align)
            self.conv4 = binarized_modules.TernarizeConv2d(conv_thres, 64, 128, kernel_size=2, stride=1, padding='valid', align=align)

            self.fc1 = binarized_modules.TernarizeLinear(first_sparsity, 1152, 256, align=align)
            self.fc2 = binarized_modules.TernarizeLinear(rest_sparsity, 256, 2, align=align)
            self.fc3 = binarized_modules.TernarizeLinear(rest_sparsity, 256, 4, align=align)
            self.fc4 = binarized_modules.TernarizeLinear(rest_sparsity, 256, 10, align=align)
        """

        self.softmax1 = torch.nn.Softmax(dim = 1)

        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=(1, 1)) #TODO: Find the padding for "SAME"
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=(0, 0))
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=(0, 0))

        self.htanh1 = nn.Hardtanh(inplace=True)
        self.htanh2 = nn.Hardtanh(inplace=True)
        self.htanh3 = nn.Hardtanh(inplace=True)
        self.htanh4 = nn.Hardtanh(inplace=True)
        self.htanh5 = nn.Hardtanh(inplace=True)
        self.htanh6 = nn.Hardtanh(inplace=True)
        self.htanh7 = nn.Hardtanh(inplace=True)
        
    def forward(self, x):
        if self.full:
            x = F.relu(self.conv1(x))
            x = self.pool1(x)
            x = F.relu(self.conv2(x))
            x = self.pool2(x)
            x = F.relu(self.conv3(x))
            x = self.pool3(x)
            x = F.relu(self.conv4(x))
            x = torch.flatten(x, start_dim=1)

            x = F.relu(self.fc1(x))
            
            out1 = self.softmax1(self.fc2(x))
            out2 = self.fc3(x)
            out3 = self.fc4(x)
        else:
            """
            if(torch.any(torch.tensor([torch.all(x[index]==0) for index in range(len(x))]))):
                #print("Label 1", self.exc_count, torch.tensor([torch.all(x[index]==0) for index in range(len(x))]))
                self.exc_count1 += int(torch.sum(torch.tensor([torch.all(x[index]==0) for index in range(len(x))])))
                #raise Exception("Stop 1")
            self.total_count += len(x)
            """

            x = self.conv1(x)
            x = self.pool1(x)
            self.htanh1(x)
            x = self.conv2(x)
            x = self.pool2(x)
            self.htanh2(x)
            x = self.conv3(x)
            x = self.pool3(x)
            self.htanh3(x)
            x = self.conv4(x)
            self.htanh4(x)
            x = torch.flatten(x, start_dim=1)

            x = F.relu(self.fc1(x))
            
            out1 = self.fc2(x)
            #self.htanh5(out1)
            out1 = self.softmax1(out1)
            out2 = self.fc3(x)
            self.htanh6(out2)
            out3 = self.fc4(x)
            self.htanh7(out3)

        return out1, out2, out3

class FC_small(nn.Module):
    def __init__(self, full=False, binary=True, first_sparsity=0.8, rest_sparsity=0.9, hid=512, ind=784, align=False):
        super(FC_small, self).__init__()
        self.align = align
        self.pruned = False
        self.hid = hid
        self.ind = ind
        
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
    