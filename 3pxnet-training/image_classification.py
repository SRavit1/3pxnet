"""# Imports"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import os
import onnx
import onnxruntime
import fnmatch

from tensorflow.keras.utils import to_categorical

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

log = open("log.txt", "w")

print("DEVICE", device)
log.write("DEVICE " + str(device) + "\n")


"""# Loading Dataset"""

def unpickle(file):
    """load the cifar-10 data"""

    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def load_cifar_10_data(data_dir, negatives=False):
    """
    Return train_data, train_filenames, train_labels, test_data, test_filenames, test_labels
    """

    # get the meta_data_dict
    # num_cases_per_batch: 1000
    # label_names: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # num_vis: :3072

    meta_data_dict = unpickle(data_dir + "/batches.meta")
    cifar_label_names = meta_data_dict[b'label_names']
    cifar_label_names = np.array(cifar_label_names)

    # training data
    cifar_train_data = None
    cifar_train_filenames = []
    cifar_train_labels = []

    for i in range(1, 6):
        cifar_train_data_dict = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            cifar_train_data = cifar_train_data_dict[b'data']
        else:
            cifar_train_data = np.vstack((cifar_train_data, cifar_train_data_dict[b'data']))
        cifar_train_filenames += cifar_train_data_dict[b'filenames']
        cifar_train_labels += cifar_train_data_dict[b'labels']

    cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
    if negatives:
        cifar_train_data = cifar_train_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_train_data = np.rollaxis(cifar_train_data, 1, 4)
    cifar_train_filenames = np.array(cifar_train_filenames)
    cifar_train_labels = np.array(cifar_train_labels)

    cifar_test_data_dict = unpickle(data_dir + "/test_batch")
    cifar_test_data = cifar_test_data_dict[b'data']
    cifar_test_filenames = cifar_test_data_dict[b'filenames']
    cifar_test_labels = cifar_test_data_dict[b'labels']

    cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3, 32, 32))
    if negatives:
        cifar_test_data = cifar_test_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)
    cifar_test_filenames = np.array(cifar_test_filenames)
    cifar_test_labels = np.array(cifar_test_labels)

    return cifar_train_data, cifar_train_filenames, to_categorical(cifar_train_labels), \
        cifar_test_data, cifar_test_filenames, to_categorical(cifar_test_labels), cifar_label_names

class FaceDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        assert len(images) == len(labels)
        self.images = images
        self.labels = labels
        self.len = len(images)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        images = self.images[index].astype(np.float32)
        labels = self.labels[index]

        labels = np.argmax(labels)
        images = np.transpose(images, (2, 0, 1))
        
        #images are in NCHW
        return images, labels

def transformImageTorch(image):
  image = np.transpose(image, (0, 3, 1, 2))
  image = image.astype(np.float32)
  image = torch.tensor(image).to(device)
  return image

import torchvision
import torchvision.transforms as transforms
def load_dataset(dataset):
   if dataset == 'CIFAR10':
      transform = transforms.Compose([transforms.ToTensor()])
      trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                              download=True, transform=transforms.Compose([transforms.RandomCrop(32, 4),
                                                                                           transforms.RandomHorizontalFlip(),
                                                                                           transforms.ToTensor()]))
      testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                             download=True, transform=transform)
      classes = ('plane', 'car', 'bird', 'cat',
                 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
   return trainset, testset, classes

BS = 32

cifar_10_dir = 'cifar-10-batches-py'
train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names = load_cifar_10_data(cifar_10_dir)

cutoff = 1000
test_data = test_data[:cutoff]
test_filenames = test_filenames[:cutoff]
test_labels = test_labels[:cutoff]

for i in range(len(label_names)):
  label_names[i] = label_names[i].decode('utf-8')

dataset = FaceDetectionDataset(train_data, train_labels)
dataset_loader = DataLoader(dataset, batch_size=BS, shuffle=True)

test_dataset = FaceDetectionDataset(test_data, test_labels)
test_dataset_loader = DataLoader(test_dataset, batch_size=BS, shuffle=True)

loss_metric = nn.CrossEntropyLoss()
test_total_batches = int(np.ceil(len(test_dataset)/BS))
total_batches = int(np.ceil(len(dataset)/BS))

dataset = "CIFAR10"
trainset, testset, classes = load_dataset(dataset)
dataset_loader = torch.utils.data.DataLoader(trainset, batch_size=BS, shuffle=True, num_workers=2)
test_dataset_loader = torch.utils.data.DataLoader(testset, batch_size=BS, shuffle=False, num_workers=2)

loss_metric = nn.CrossEntropyLoss()
test_total_batches = int(np.ceil(len(testset)/BS))
total_batches = int(np.ceil(len(trainset)/BS))

"""# Creating Model

## Defining Binarized Layers
"""

def permute_from_list(mask, permute_list, transpose=False):
   # (0. Reshape for 4D tensor). 1. Divide into sections. 2. Permute individually (3. Reshape back for 4D tensor)
   permute_redund = int(permute_list.size(0))
   if len(mask.size())==4:
      mask_flat = mask.permute(0,2,3,1).contiguous().view(-1, mask.size(1)).contiguous()
   else:
      mask_flat = mask.clone()
   split_size = int(np.ceil(float(mask_flat.size(0))/permute_redund).astype(int))
   mask_unpermute = torch.zeros_like(mask_flat)
   length = permute_list.size(1)
   mask_split = torch.split(mask_flat, split_size)
   permute_redund_cor = min(permute_redund, np.ceil(float(mask.size(0))/split_size).astype(int))
   for i in range(permute_redund_cor):
      if transpose:
         permute_list_t = torch.zeros_like(permute_list[i])
         permute_list_t[permute_list[i]] = torch.arange(length).to(permute_list.device)
         mask_unpermute[i*split_size:(i+1)*split_size] = mask_split[i][:,permute_list_t]
      else:
         mask_unpermute[i*split_size:(i+1)*split_size] = mask_split[i][:,permute_list[i]]

   if len(mask.size())==4:
      mask_unpermute = mask_unpermute.view(mask.size(0), mask.size(2), mask.size(3), mask.size(1)).permute(0,3,1,2)
   return mask_unpermute

def Binarize(tensor,quant_mode='det'):
    if quant_mode=='det':
        return tensor.sign()
    if quant_mode=='bin':
        return (tensor>=0).type(type(tensor))*2-1
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)

def Ternarize(tensor, mult = 0.7, mask = None, permute_list = None, pruned = False, align = False, pack = 32):
    if type(mask) == type(None):
        mask = torch.ones_like(tensor)
    
    # Fix permutation. Tensor needs to be permuted
    if not pruned:
        tensor_masked = permute_from_list(tensor, permute_list)
        if len(tensor_masked.size())==4:
            tensor_masked = tensor_masked.permute(0,2,3,1)
       
        if not align:
            tensor_flat = torch.abs(tensor_masked.contiguous().view(-1)).contiguous()
            tensor_split = torch.split(tensor_flat, pack, dim=0)
            tensor_split = torch.stack(tensor_split, dim=0)
            tensor_sum = torch.sum(tensor_split, dim=1)
            tensor_size = tensor_sum.size(0)
            tensor_sorted, _ = torch.sort(tensor_sum)
            thres = tensor_sorted[int(mult*tensor_size)]
            tensor_flag = torch.ones_like(tensor_sum)
            tensor_flag[tensor_sum.ge(-thres) * tensor_sum.le(thres)] = 0
            tensor_flag = tensor_flag.repeat(pack).reshape(pack,-1).transpose(1,0).reshape_as(tensor_masked)
            
        else:
            tensor_flat = torch.abs(tensor_masked.reshape(tensor_masked.size(0),-1)).contiguous()
            tensor_split = torch.split(tensor_flat, pack, dim=1)
            tensor_split = torch.stack(tensor_split, dim=1)
            tensor_sum = torch.sum(tensor_split, dim=2)
            tensor_size = tensor_sum.size(1)
            tensor_sorted, _ = torch.sort(tensor_sum, dim=1)
            tensor_sorted = torch.flip(tensor_sorted, [1])
            multiplier = 32./pack
            index = int(torch.ceil((1-mult)*tensor_size/multiplier)*multiplier)
            thres = tensor_sorted[:, index-1].view(-1,1)
            tensor_flag = torch.zeros_like(tensor_sum)
            tensor_flag[tensor_sum.ge(thres)] = 1
            tensor_flag[tensor_sum.le(-thres)] = 1
            tensor_flag = tensor_flag.repeat(1,pack).reshape(tensor_flag.size(0),pack,-1).transpose(2,1).reshape_as(tensor_masked)

        if len(tensor_masked.size())==4:
            tensor_flag = tensor_flag.permute(0,3,1,2)            
        tensor_flag = permute_from_list(tensor_flag, permute_list, transpose=True)
        tensor_bin = tensor.sign() * tensor_flag
            
    else:
        tensor_bin = tensor.sign() * mask
        
    return tensor_bin 

class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)
        self.register_buffer('weight_org', self.weight.data.clone())

    def forward(self, input):

        if (input.size(1) != 784) and (input.size(1) != 3072):
            input.data=Binarize(input.data)
        self.weight.data=Binarize(self.weight_org)
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out
    
class TernarizeLinear(nn.Linear):

    def __init__(self, thres, *kargs, **kwargs):
        try:
            pack = kwargs['pack']
        except:
            pack = 32
        else:
            del(kwargs['pack'])
        try:
            permute = kwargs['permute']
        except:
            permute = 1
        else:
            del(kwargs['permute'])
        try:
            self.align=kwargs['align']
        except:
            self.align=True
        else:
            del(kwargs['align'])
        super(TernarizeLinear, self).__init__(*kargs, **kwargs)
        
        permute = min(permute, self.weight.size(0))
        self.register_buffer('pack', torch.LongTensor([pack]))
        self.register_buffer('thres', torch.FloatTensor([thres]))
        self.register_buffer('mask', torch.ones_like(self.weight.data))
        self.register_buffer('permute_list', torch.LongTensor(np.tile(range(self.weight.size(1)), (permute,1))))
        self.register_buffer('weight_org', self.weight.data.clone())

    def forward(self, input, pruned=False):

        if (input.size(1) != 784) and (input.size(1) != 3072):
            input.data=Binarize(input.data)
        self.weight.data=Ternarize(self.weight_org, self.thres, self.mask, self.permute_list, pruned, align=self.align, pack=self.pack.item())
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)
        return out

class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)
        self.register_buffer('weight_org', self.weight.data.clone())

    def forward(self, input):
        if input.size(1) != 3:
            input.data = Binarize(input.data)
        self.weight.data=Binarize(self.weight_org)

        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out
    
class TernarizeConv2d(nn.Conv2d):

    def __init__(self, thres, *kargs, **kwargs):
        try:
            pack = kwargs['pack']
        except:
            pack = 32
        else:
            del(kwargs['pack'])
        try:
            permute = kwargs['permute']
        except:
            permute = 1
        else:
            del(kwargs['permute'])
        try:
            self.align=kwargs['align']
        except:
            self.align=True
        else:
            del(kwargs['align'])
            
        super(TernarizeConv2d, self).__init__(*kargs, **kwargs)
        permute = min(permute, self.weight.size(0))
        self.register_buffer('pack', torch.LongTensor([pack]))
        self.register_buffer('thres', torch.FloatTensor([thres]))
        self.register_buffer('mask', torch.ones_like(self.weight.data))
        self.register_buffer('permute_list', torch.LongTensor(np.tile(range(self.weight.size(1)), (permute,1))))
        self.register_buffer('weight_org', self.weight.data.clone())

    def forward(self, input, pruned=False):
        if input.size(1) != 3:
            input.data = Binarize(input.data)
        self.weight.data=Ternarize(self.weight_org, self.thres, self.mask, self.permute_list, pruned, align=self.align, pack=self.pack.item())
        out = nn.functional.conv2d(input, self.weight, None, self.stride, self.padding, self.dilation, self.groups)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out

def adjust_pack(net, pack):
   for mod in net.modules():
      if hasattr(mod, 'pack'):
         mod.pack -= mod.pack - pack

def one_time_permute(weight_gpu, thres, pack=8, weight_grad_gpu=None):
   '''
   Form the permutation list of a weight tensor given a sparsity constraint
   '''
   permute_list = list(range(weight_gpu.size(1)))
   permute_size = 0
   weight = (weight_gpu * weight_gpu)
   if type(weight_grad_gpu)!=type(None):
      weight_grad = weight_grad_gpu.cpu()
      grad_unprune = weight_grad * (weight==1).type(torch.FloatTensor) - weight_grad * (weight==-1).type(torch.FloatTensor) + abs(weight_grad) * (weight==0).type(torch.FloatTensor)
   # 1. Find the n-pack that has the maximum overlap
   counter = 0
   start_time = time.time()
   while permute_size+pack < weight.size(1):
      permute_list_valid = permute_list[permute_size:]

      result_tensor = torch.zeros((weight.size(0), pack)).to(weight_gpu.device)
      start_tensor = weight[:, permute_list_valid[0]]
      if type(weight_grad)!=type(None):
         start_gradient = grad_unprune[:, permute_list_valid[0]]
         result_gradient = torch.zeros((weight.size(0), pack))
         result_gradient[:,0] = start_gradient
      result_tensor[:,0] = start_tensor
      current_permute = [permute_list_valid[0]]
      permute_list_valid.remove(permute_list_valid[0])
      for i in range(1, pack, 1):
         max_score = -100000000
         max_index = -1
         for index in permute_list_valid:
            score_weight = sim_n(result_tensor[:,:i], weight[:,index])
            score = sim_n_grad(result_gradient[:,:i], grad_unprune[:,index])/100
            score = score_weight + score
            if score > max_score:
               max_score = score
               max_index = index
         result_tensor[:,i] = weight[:,max_index]
         result_gradient[:,i] = grad_unprune[:,max_index]
         permute_list_valid.remove(max_index)
         current_permute.append(max_index)

      # 2. Form permutation list such that these n columns are in left-most positions
      permute_list_finished = permute_list[:permute_size] + current_permute
      permute_list = permute_list_finished + [item for item in permute_list if item not in permute_list_finished]
      permute_size += pack
      counter += 1
   return torch.LongTensor(permute_list).to(weight_gpu.device)

def perm_sort(weight):
   '''
   Form permutation list based on the RMS value of each column
   '''
   w_sum = torch.sum(torch.abs(weight), dim = 0)
   permute_list = np.argsort(w_sum.detach().data.cpu().numpy())
   permute_list = np.ascontiguousarray(np.flipud(permute_list))
   return torch.from_numpy(permute_list).type('torch.LongTensor').to(weight.device)

def sim_n(tensor_group, tensor_new):
   '''
   Compute similarity score of a set of tensors and a new tensor
   '''
   height = tensor_group.size(0)
   width = tensor_group.size(1)+1
   tensor_group_new = torch.zeros((height, width)).to(tensor_group.device)
   tensor_group_new[:,:(width-1)] = tensor_group
   tensor_group_new[:,width-1] = tensor_new
   empty = torch.sum((tensor_group_new==0), dim=1).to(device=tensor_group.device, dtype=torch.float)
   compressed = torch.sum(tensor_group_new, dim=1)
   tensor_sum = -torch.sum(torch.min(empty, compressed))
   return tensor_sum

def sim_n_grad(grad_group, grad_new):
   raw_score = torch.sum(grad_group, dim=1)+grad_new
   score = -torch.abs(raw_score)
   tensor_sum = torch.sum(score)
   return tensor_sum

def permute_all_weights_once(model, pack=8, mode=1):
   '''
   Determine permutation list of all modules of a network without pruning
   '''
   # Only permute. Pruning is done using something else
   import traceback
   for mod in model.modules():
      try:
         if isinstance(mod, nn.Linear):
            #logging.info('Permuting '+ str(mod))
            cur_pack = pack
            permute_redund = mod.permute_list.size(0)
            section_size = np.ceil(float(mod.weight.size(0))/permute_redund).astype(int)
            permute_redund_cor = min(permute_redund, np.ceil(float(mod.weight.size(0))/section_size).astype(int))
            for i in range(permute_redund_cor):
               ceiling = min((i+1)*section_size, mod.weight.size(0))
               if mode==1:
                  mod.permute_list[i] = one_time_permute(mod.weight.data[i*section_size:ceiling], mod.thres, pack=cur_pack, weight_grad_gpu = mod.weight.grad[i*section_size:(i+1)*section_size])
               elif mode==0:
                  mod.permute_list[i] = perm_sort(mod.weight_org[i*section_size:ceiling])
               elif mode==2: #I think perm_rand isn't defined here
                  mod.permute_list[i] = perm_rand(mod.weight.data[i*section_size:ceiling])
         elif isinstance(mod, nn.Conv2d):
            #logging.info('Permuting '+ str(mod))
            weight_flat = mod.weight.data.permute(0,2,3,1).contiguous().view(-1,mod.weight.size(1)).contiguous()
            grad_flat = mod.weight.grad.permute(0,2,3,1).contiguous().view(-1,mod.weight.size(1)).contiguous()
            if mode==1:
               mod.permute_list[0] = one_time_permute(weight_flat, mod.thres, pack=pack, weight_grad_gpu = grad_flat)
            elif mode==1: #Should this be mode==0?
               mod.permute_list[0] = perm_sort(weight_flat)
            elif mode==2:
               mod.permute_list[0] = perm_rand(weight_flat)
      except:
         traceback.print_exc()

"""## Defining Pytorch Model

### Primary Model
"""
#Resnet
class ImageClassifier(nn.Module):
    def __init__(self, full=True, binary=True, conv_thres=0.1, lin_sparsity=0.2, align=False):
      super(ImageClassifier, self).__init__()
      self.align = align
      self.pruned = False
      self.full = full
      self.binary = binary

      num_filters = 16
      num_filters_2 = 32
      num_filters_3 = 64

      if full:
        self.conv1 = nn.Conv2d(3, num_filters, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(num_filters, num_filters_2, kernel_size=3, stride=2)
        self.conv5 = nn.Conv2d(num_filters_2, num_filters_2, kernel_size=3, stride=1)
        self.conv6 = nn.Conv2d(num_filters, num_filters_2, kernel_size=1, stride=2)
        self.conv7 = nn.Conv2d(num_filters_2, num_filters_3, kernel_size=3, stride=2)
        self.conv8 = nn.Conv2d(num_filters_3, num_filters_3, kernel_size=3, stride=1)
        self.conv9 = nn.Conv2d(num_filters_2, num_filters_3, kernel_size=1, stride=2)
        
        self.fc1 = nn.Linear(64, 10)
      elif binary:
        self.conv1 = nn.Conv2d(3, num_filters, kernel_size=3, stride=1)
        self.conv2 = BinarizeConv2d(num_filters, num_filters, kernel_size=3, stride=1)
        self.conv3 = BinarizeConv2d(num_filters, num_filters, kernel_size=3, stride=1)
        self.conv4 = BinarizeConv2d(num_filters, num_filters_2, kernel_size=3, stride=2)
        self.conv5 = BinarizeConv2d(num_filters_2, num_filters_2, kernel_size=3, stride=1)
        self.conv6 = BinarizeConv2d(num_filters, num_filters_2, kernel_size=1, stride=2)
        self.conv7 = BinarizeConv2d(num_filters_2, num_filters_3, kernel_size=3, stride=2)
        self.conv8 = nn.Conv2d(num_filters_3, num_filters_3, kernel_size=3, stride=1)
        self.conv9 = nn.Conv2d(num_filters_2, num_filters_3, kernel_size=1, stride=2)
        
        self.fc1 = nn.Linear(64, 10)
      else:
        self.conv1 = nn.Conv2d(3, num_filters, kernel_size=3, stride=1)
        self.conv2 = TernarizeConv2d(conv_thres, num_filters, num_filters, kernel_size=3, stride=1, align=align)
        self.conv3 = TernarizeConv2d(conv_thres, num_filters, num_filters, kernel_size=3, stride=1, align=align)
        self.conv4 = TernarizeConv2d(conv_thres, num_filters, num_filters_2, kernel_size=3, stride=2, align=align)
        self.conv5 = TernarizeConv2d(conv_thres, num_filters_2, num_filters_2, kernel_size=3, stride=1, align=align)
        self.conv6 = TernarizeConv2d(conv_thres, num_filters, num_filters_2, kernel_size=1, stride=2, align=align)
        self.conv7 = TernarizeConv2d(conv_thres, num_filters_2, num_filters_3, kernel_size=3, stride=2, align=align)
        self.conv8 = nn.Conv2d(num_filters_3, num_filters_3, kernel_size=3, stride=1)
        self.conv9 = nn.Conv2d(num_filters_2, num_filters_3, kernel_size=1, stride=2)
        
        self.fc1 = nn.Linear(64, 10)

      self.batchnorm1 = nn.BatchNorm2d(num_filters)
      self.batchnorm2 = nn.BatchNorm2d(num_filters)
      self.batchnorm3 = nn.BatchNorm2d(num_filters)
      self.batchnorm4 = nn.BatchNorm2d(num_filters_2)
      self.batchnorm5 = nn.BatchNorm2d(num_filters_2)
      self.batchnorm6 = nn.BatchNorm2d(num_filters_2)
      self.batchnorm7 = nn.BatchNorm2d(num_filters_3)
      self.batchnorm8 = nn.BatchNorm2d(num_filters_3)
      self.batchnorm9 = nn.BatchNorm2d(num_filters_3)
      self.batchnorm10 = nn.BatchNorm1d(10)

      self.pool1 = nn.MaxPool2d((5, 5), stride=1)
      self.pool2 = nn.MaxPool2d((4, 4), stride=1)
      self.pool3 = nn.MaxPool2d((2, 2))

      self.act = F.relu if self.full else nn.Hardtanh()
    def forward(self, x):
      """
      x_temp = x.cpu().detach().numpy()
      x_temp = np.transpose(x_temp, (0, 3, 2, 1))
      print("pool1_out output", x_temp.shape, x_temp.flatten()[:10])
      """

      #x = F.pad(x, (1, 1, 1, 1))
      x = self.conv1(x)
      x = self.batchnorm1(x)
      x = self.act(x)

      # First stack
      #y = F.pad(x, (1, 1, 1, 1))
      y = x
      y = self.conv2(y)
      y = self.batchnorm2(y)
      y = self.act(y)

      #y = F.pad(y, (1, 1, 1, 1))
      y = self.conv3(y)
      y = self.batchnorm3(y)

      x = self.pool1(x)

      x = x + y
      x = self.act(x)

      # Second stack
      #y = F.pad(x, (0, 1, 0, 1))
      y = x
      y = self.conv4(y)
      y = self.batchnorm4(y)
      y = self.act(y)

      #y = F.pad(y, (1, 1, 1, 1))
      y = self.conv5(y)
      y = self.batchnorm5(y)

      x = self.conv6(x)
      x = self.batchnorm6(x)

      x = self.pool2(x)

      x = x + y
      x = self.act(x)

      # Third stack
      #y = F.pad(x, (0, 1, 0, 1))
      y = x
      y = self.conv7(y)
      y = self.batchnorm7(y)
      y = self.act(y)

      #y = F.pad(y, (1, 1, 1, 1))
      y = self.conv8(y)
      y = self.batchnorm8(y)

      x = self.conv9(x)
      x = self.batchnorm9(y)

      x = x + y
      x = self.act(x)

      # Final classification layer.
      x = self.pool3(x)

      y = torch.squeeze(x)
      x = self.fc1(y)
      x = self.batchnorm10(x)

      return x

"""### Simple Model"""

#"Resnet" without residuals-- sequential model
class ImageClassifierSimple(nn.Module):
    def __init__(self, full=True, binary=True, conv_thres=0.1, lin_sparsity=0.2, align=False):
      super(ImageClassifierSimple, self).__init__()
      self.align = align
      self.pruned = False
      self.full = full
      self.binary = binary

      num_filters = 16
      num_filters_2 = 32
      num_filters_3 = 64

      if full:
        self.conv1 = nn.Conv2d(3, num_filters, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters_2, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(num_filters_2, num_filters_2, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(num_filters_2, num_filters_3, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(num_filters_3, num_filters_3, kernel_size=3, stride=1)
        
        self.fc1 = nn.Linear(64, 10, bias=False)
      elif binary:
        self.conv1 = nn.Conv2d(3, num_filters, kernel_size=3, stride=1)
        self.conv2 = BinarizeConv2d(num_filters, num_filters_2, kernel_size=3, stride=1, bias=False)
        self.conv3 = BinarizeConv2d(num_filters_2, num_filters_2, kernel_size=3, stride=1, bias=False)
        self.conv4 = BinarizeConv2d(num_filters_2, num_filters_3, kernel_size=3, stride=1, bias=False)
        self.conv5 = BinarizeConv2d(num_filters_3, num_filters_3, kernel_size=3, stride=1, bias=False)
        
        self.fc1 = nn.Linear(64, 10, bias=False)
      else:
        self.conv1 = nn.Conv2d(3, num_filters, kernel_size=3, stride=1)
        self.conv2 = TernarizeConv2d(conv_thres, num_filters, num_filters_2, kernel_size=3, stride=1, align=align, bias=False)
        self.conv3 = TernarizeConv2d(conv_thres, num_filters_2, num_filters_2, kernel_size=3, stride=1, align=align, bias=False)
        self.conv4 = TernarizeConv2d(conv_thres, num_filters_2, num_filters_3, kernel_size=3, stride=1, align=align, bias=False)
        self.conv5 = TernarizeConv2d(conv_thres, num_filters_3, num_filters_3, kernel_size=3, stride=1, align=align, bias=False)
        
        self.fc1 = nn.Linear(64, 10, bias=False)

      self.batchnorm1 = nn.BatchNorm2d(num_filters)
      self.batchnorm2 = nn.BatchNorm2d(num_filters_2)
      self.batchnorm3 = nn.BatchNorm2d(num_filters_2)
      self.batchnorm4 = nn.BatchNorm2d(num_filters_3)
      self.batchnorm5 = nn.BatchNorm2d(num_filters_3)
      self.batchnorm6 = nn.BatchNorm1d(10)

      self.act = F.relu if self.full else nn.Hardtanh()

      self.pool1 = nn.MaxPool2d((3, 3))
      self.pool2 = nn.MaxPool2d((3, 3))
    def forward(self, x):
      x = self.conv1(x)
      x = self.batchnorm1(x)
      x = self.act(x)
      x = self.conv2(x)
      x = self.batchnorm2(x)
      x = self.act(x)
      x = self.conv3(x)
      x = self.batchnorm3(x)
      x = self.act(x)
      x = self.pool1(x)
      x = self.conv4(x)
      x = self.batchnorm4(x)
      x = self.act(x)
      x = self.conv5(x)
      x = self.batchnorm5(x)
      x = self.act(x)
      x = self.pool2(x)

      #x = torch.flatten(x, start_dim=1)
      x = torch.squeeze(x)
      x = self.fc1(x)
      x = self.batchnorm6(x)
      x = self.act(x)

      return x

"""### CNN_Medium"""
# Provided with 3pxnet examples
class CNN_medium(nn.Module):

    def __init__(self, full=False, binary=True, conv_thres=0.7, fc_thres=0.9, align=False, pad=0):
        super(CNN_medium, self).__init__()
        
        self.pruned = False
        self.full = full
        self.binary = binary
        self.pad = pad
        
        self.conv1 = BinarizeConv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.htanh1 = nn.Hardtanh(inplace=True)
        
        if full:
            self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
            self.fc1 = nn.Linear(512*4*4, 10, bias=False)
        elif binary:
            self.conv1 = BinarizeConv2d(3, 128, kernel_size=3, stride=1, padding=0, bias=False)
            self.conv2 = BinarizeConv2d(128, 128, kernel_size=3, stride=1, padding=0, bias=False)
            self.conv3 = BinarizeConv2d(128, 256, kernel_size=3, stride=1, padding=0, bias=False)
            self.conv4 = BinarizeConv2d(256, 256, kernel_size=3, stride=1, padding=0, bias=False)
            self.conv5 = BinarizeConv2d(256, 512, kernel_size=3, stride=1, padding=0, bias=False)
            self.conv6 = BinarizeConv2d(512, 512, kernel_size=3, stride=1, padding=0, bias=False)
            self.conv7 = BinarizeConv2d(512, 10, kernel_size=4, padding=0, bias=False)
            self.fc1 = BinarizeLinear(1024, 1024, bias=False)
        else:
            self.conv1 = BinarizeConv2d(3, 128, kernel_size=3, stride=1, padding=0, bias=False)
            self.conv2 = TernarizeConv2d(conv_thres, 128, 128, kernel_size=3, padding=0, bias=False, align=align)
            self.conv3 = TernarizeConv2d(conv_thres, 128, 256, kernel_size=3, padding=0, bias=False, align=align)
            self.conv4 = TernarizeConv2d(conv_thres, 256, 256, kernel_size=3, padding=0, bias=False, align=align)
            self.conv5 = TernarizeConv2d(conv_thres, 256, 512, kernel_size=3, padding=0, bias=False, align=align)
            self.conv6 = TernarizeConv2d(conv_thres, 512, 512, kernel_size=3, padding=0, bias=False, align=align)
            self.conv7 = TernarizeConv2d(0.49, 512, 10, kernel_size=4, padding=0, bias=False, align=align)
            self.fc1 = BinarizeLinear(1024, 1024, bias=False)
            
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
            x = self.htanh1(x)
            
            x = F.pad(x, (1,1,1,1), value=self.pad)
            if self.binary:
                x = self.conv2(x)
            else:
                x = self.conv2(x, self.pruned)
            x = self.pool2(x)
            x = self.bn2(x)
            x = self.htanh2(x)
            
            x = F.pad(x, (1,1,1,1), value=self.pad)
            if self.binary:
                x = self.conv3(x)
            else:
                x = self.conv3(x, self.pruned)
            x = self.bn3(x)
            x = self.htanh3(x)
            
            x = F.pad(x, (1,1,1,1), value=self.pad)
            if self.binary:
                x = self.conv4(x)
            else:
                x = self.conv4(x, self.pruned)
            x = self.pool4(x)
            x = self.bn4(x)
            x = self.htanh4(x)
            
            x = F.pad(x, (1,1,1,1), value=self.pad)
            if self.binary:
                x = self.conv5(x)
            else:
                x = self.conv5(x, self.pruned)
            x = self.bn5(x)
            x = self.htanh5(x)
            
            x = F.pad(x, (1,1,1,1), value=self.pad)
            if self.binary:
                x = self.conv6(x)
            else:
                x = self.conv6(x, self.pruned)
            x = self.pool6(x)
            x = self.bn6(x)
            x = self.htanh6(x)
            
            if self.binary:
                x = self.conv7(x)
            else:
                x = self.conv7(x, self.pruned)
            x = x.view(-1, 10)
            x = self.bnfc1(x)
        return self.logsoftmax(x)

"""### CNN_medium_modified"""
# Smaller version of CNN_Medium
class CNN_medium_modified(nn.Module):

    def __init__(self, full=False, binary=True, conv_thres=0.7, fc_thres=0.9, align=False, pad=0):
        super(CNN_medium_modified, self).__init__()
        
        self.pruned = False
        self.full = full
        self.binary = binary
        self.pad = pad
        
        self.no_filters_1 = 32
        self.no_filters_2 = 32
        self.no_filters_3 = 64

        self.conv1 = BinarizeConv2d(3, self.no_filters_1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.no_filters_1)
        self.htanh1 = nn.Hardtanh(inplace=True)
        
        if full:
            self.conv1 = nn.Conv2d(3, self.no_filters_1, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv2 = nn.Conv2d(self.no_filters_1, self.no_filters_1, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv3 = nn.Conv2d(self.no_filters_1, self.no_filters_2, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv4 = nn.Conv2d(self.no_filters_2, self.no_filters_2, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv5 = nn.Conv2d(self.no_filters_2, self.no_filters_3, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv6 = nn.Conv2d(self.no_filters_3, self.no_filters_3, kernel_size=3, stride=1, padding=1, bias=False)
            self.fc1 = nn.Linear(self.no_filters_3*4*4, 10, bias=False)
        elif binary:
            """
            self.conv1 = nn.Conv2d(3, self.no_filters_1, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv2 = BinarizeConv2d(self.no_filters_1, self.no_filters_1, kernel_size=3, stride=1, padding=0, bias=False)
            self.conv3 = BinarizeConv2d(self.no_filters_1, self.no_filters_2, kernel_size=3, stride=1, padding=0, bias=False)
            self.conv4 = BinarizeConv2d(self.no_filters_2, self.no_filters_2, kernel_size=3, stride=1, padding=0, bias=False)
            self.conv5 = BinarizeConv2d(self.no_filters_2, self.no_filters_3, kernel_size=3, stride=1, padding=0, bias=False)
            self.conv6 = BinarizeConv2d(self.no_filters_3, self.no_filters_3, kernel_size=3, stride=1, padding=0, bias=False)
            self.conv7 = BinarizeConv2d(self.no_filters_3, 10, kernel_size=4, padding=0, bias=False)
            self.fc1 = nn.Linear(1024, 1024, bias=False) #BinarizeLinear(1024, 1024, bias=False)
            """
            self.conv1 = nn.Conv2d(3, self.no_filters_1, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv2 = BinarizeConv2d(self.no_filters_1, self.no_filters_1, kernel_size=3, stride=1, padding=0, bias=False)
            self.conv3 = BinarizeConv2d(self.no_filters_1, self.no_filters_2, kernel_size=3, stride=1, padding=0, bias=False)
            self.conv4 = BinarizeConv2d(self.no_filters_2, self.no_filters_2, kernel_size=3, stride=1, padding=0, bias=False)
            self.conv5 = BinarizeConv2d(self.no_filters_2, self.no_filters_3, kernel_size=3, stride=1, padding=0, bias=False)
            self.conv6 = BinarizeConv2d(self.no_filters_3, self.no_filters_3, kernel_size=3, stride=1, padding=0, bias=False)
            self.conv7 = BinarizeConv2d(self.no_filters_3, 10, kernel_size=4, padding=0, bias=False)
            self.fc1 = nn.Linear(1024, 1024, bias=False) #BinarizeLinear(1024, 1024, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, self.no_filters_1, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv2 = TernarizeConv2d(conv_thres, self.no_filters_1, self.no_filters_1, kernel_size=3, padding=0, bias=False, align=align)
            self.conv3 = TernarizeConv2d(conv_thres, self.no_filters_1, self.no_filters_2, kernel_size=3, padding=0, bias=False, align=align)
            self.conv4 = TernarizeConv2d(conv_thres, self.no_filters_2, self.no_filters_2, kernel_size=3, padding=0, bias=False, align=align)
            self.conv5 = TernarizeConv2d(conv_thres, self.no_filters_2, self.no_filters_3, kernel_size=3, padding=0, bias=False, align=align)
            self.conv6 = TernarizeConv2d(conv_thres, self.no_filters_3, self.no_filters_3, kernel_size=3, padding=0, bias=False, align=align)
            self.conv7 = TernarizeConv2d(0.49, self.no_filters_3, 10, kernel_size=4, padding=0, bias=False, align=align)
            self.fc1 = nn.Linear(1024, 1024, bias=False) #BinarizeLinear(1024, 1024, bias=False)
            
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(self.no_filters_1)
        self.htanh2 = nn.Hardtanh(inplace=True)
        
        
        self.bn3 = nn.BatchNorm2d(self.no_filters_2)
        self.htanh3 = nn.Hardtanh(inplace=True)
        
        
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn4 = nn.BatchNorm2d(self.no_filters_2)
        self.htanh4 = nn.Hardtanh(inplace=True)
        
        
        self.bn5 = nn.BatchNorm2d(self.no_filters_3)
        self.htanh5 = nn.Hardtanh(inplace=True)
        
        
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn6 = nn.BatchNorm2d(self.no_filters_3)
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
            
            x = x.view(-1, self.no_filters_3*4*4)
            x = F.relu(self.fc1(x))
            self.fc1_result = x.data.clone()
        else:
            x = F.pad(x, (1,1,1,1), value=self.pad)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.htanh1(x)
            
            x = F.pad(x, (1,1,1,1), value=self.pad)
            if self.binary:
                x = self.conv2(x)
            else:
                x = self.conv2(x, self.pruned)
            x = self.pool2(x)
            x = self.bn2(x)
            x = self.htanh2(x)
            
            x = F.pad(x, (1,1,1,1), value=self.pad)
            if self.binary:
                x = self.conv3(x)
            else:
                x = self.conv3(x, self.pruned)
            x = self.bn3(x)
            x = self.htanh3(x)
            
            x = F.pad(x, (1,1,1,1), value=self.pad)
            if self.binary:
                x = self.conv4(x)
            else:
                x = self.conv4(x, self.pruned)
            x = self.pool4(x)
            x = self.bn4(x)
            x = self.htanh4(x)
            
            x = F.pad(x, (1,1,1,1), value=self.pad)
            if self.binary:
                x = self.conv5(x)
            else:
                x = self.conv5(x, self.pruned)
            x = self.bn5(x)
            x = self.htanh5(x)
            
            x = F.pad(x, (1,1,1,1), value=self.pad)
            if self.binary:
                x = self.conv6(x)
            else:
                x = self.conv6(x, self.pruned)
            x = self.pool6(x)
            x = self.bn6(x)
            x = self.htanh6(x)
            
            if self.binary:
                x = self.conv7(x)
            else:
                x = self.conv7(x, self.pruned)
            x = x.view(-1, 10)
            x = self.bnfc1(x)
        return self.logsoftmax(x)

"""# Training"""

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

classifier = ImageClassifierSimple

full_model = classifier(True).to(device)
binarized_model = classifier(False, True).to(device)
ternarized_model = classifier(False, False, 0.7, 0.8).to(device)

models = [("full", full_model), ("binarized", binarized_model), ("ternarized", ternarized_model)]
#models = [models[2]]

__optimizers = {
    'SGD': torch.optim.SGD,
    'ASGD': torch.optim.ASGD,
    'Adam': torch.optim.Adam,
    'Adamax': torch.optim.Adamax,
    'Adagrad': torch.optim.Adagrad,
    'Adadelta': torch.optim.Adadelta,
    'Rprop': torch.optim.Rprop,
    'RMSprop': torch.optim.RMSprop
}

def adjust_optimizer(optimizer, epoch, config):
    """Reconfigures the optimizer according to epoch and config dict"""
    def modify_optimizer(optimizer, setting):
        if 'optimizer' in setting:
            optimizer = __optimizers[setting['optimizer']](optimizer.param_groups)
            print('OPTIMIZER - setting method = %s' %
                          setting['optimizer'])
        for param_group in optimizer.param_groups:
            for key in param_group.keys():
                if key in setting:
                    print('OPTIMIZER - setting %s = %s' %
                                  (key, setting[key]))
                    param_group[key] = setting[key]
        return optimizer

    if callable(config):
        optimizer = modify_optimizer(optimizer, config(epoch))
    else:
        for e in range(epoch + 1):  # run over all epochs - sticky setting
            if e in config:
                optimizer = modify_optimizer(optimizer, config[e])

    return optimizer

def train(model, optimizer, dataset_loader, test_dataset_loader, loss_metric, EPOCHS, history):
  regime = getattr(model, 'regime', {0: {'optimizer': 'Adam', 'lr': 0.1,'momentum': 0.9,'weight_decay': 1e-4}})
  for epoch in range(EPOCHS):
    #optimizer = adjust_optimizer(optimizer, epoch, regime)
    batch_no = 0

    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()

    test_loss_meter = AverageMeter()
    test_accuracy_meter = AverageMeter()
    
    model.train()
    for images, labels in dataset_loader:
      print("\rEpoch {0} Batch: {1}/{2} Loss: {3:.4f} Accuracy: {4:.4f}".format(epoch, batch_no, total_batches, loss_meter.avg, accuracy_meter.avg), end="")
      log.write("\rEpoch {0} Batch: {1}/{2} Loss: {3:.4f} Accuracy: {4:.4f}".format(epoch, batch_no, total_batches, loss_meter.avg, accuracy_meter.avg))
      log.flush()
      
      images = images.to(device)
      labels = labels.to(device)

      no_images = images.shape[0]
      labels_pred = model.forward(images)
      loss = loss_metric(labels_pred, labels)
      accuracy = torch.sum(torch.argmax(labels_pred, dim=1) == labels) / no_images

      optimizer.zero_grad()
      loss.backward()
      for p in model.modules():
        if hasattr(p, 'weight_org'):
            p.weight.data.copy_(p.weight_org)
      optimizer.step()
      for p in model.modules():
        if hasattr(p, 'weight_org'):
            p.weight_org.copy_(p.weight.data.clamp_(-1,1))

      loss_meter.update(loss, no_images)
      accuracy_meter.update(accuracy, no_images)
      batch_no += 1

    model.eval()
    for images, labels in test_dataset_loader:
      images = images.to(device)
      labels = labels.to(device)

      no_images = images.shape[0]
      with torch.no_grad():
        labels_pred = model.forward(images)
      loss = loss_metric(labels_pred, labels)
      accuracy = torch.sum(torch.argmax(labels_pred, dim=1) == labels) / no_images

      test_loss_meter.update(loss, no_images)
      test_accuracy_meter.update(accuracy, no_images)

    print("\rEpoch {0} Loss: {1:.4f} Accuracy: {2:.4f} Test loss: {3:.4f} Test accuracy: {4:.4f}".format(epoch, 
      loss_meter.avg, accuracy_meter.avg, test_loss_meter.avg, test_accuracy_meter.avg))
    log.write("\rEpoch {0} Loss: {1:.4f} Accuracy: {2:.4f} Test loss: {3:.4f} Test accuracy: {4:.4f}\n".format(epoch, 
      loss_meter.avg, accuracy_meter.avg, test_loss_meter.avg, test_accuracy_meter.avg))
    log.flush()

    history['loss'].append(float(loss_meter.avg))
    history['accuracy'].append(float(accuracy_meter.avg))
    history['test loss'].append(float(test_loss_meter.avg))
    history['test accuracy'].append(float(test_accuracy_meter.avg))

EPOCHS = 25
for (modelName, model) in models:
  pack = 32
  permute = 1

  print("Begin train", modelName)
  log.write("Begin train " + str(modelName) + "\n")
  log.flush()

  learning_rate = 1e-4
  lr_decay = np.power((2e-6/learning_rate), (1./100))
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  #scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_decay)

  history = {'loss': [], 'accuracy': [], 'test loss': [], 'test accuracy': []}
  #train(model, optimizer, dataset_loader, test_dataset_loader, loss_metric, EPOCHS, history)

  # Retrain with permutation + packing constraint
  adjust_pack(model, pack)
  permute_all_weights_once(model, pack=pack, mode=permute)

  #train(model, optimizer, dataset_loader, test_dataset_loader, loss_metric, EPOCHS, history)

  # Fix pruned packs and fine tune
  for mod in model.modules():
      if hasattr(mod, 'mask'):
          mod.mask = torch.abs(mod.weight.data)    
  model.pruned = True

  torch.save(model, "trained_models/" + modelName + ".pt")

  #train(model, optimizer, dataset_loader, test_dataset_loader, loss_metric, 200, history)

  epochs_total = EPOCHS + EPOCHS + 200

  """
  plt.plot(np.array(range(epochs_total)), history['loss'], label='loss')
  plt.plot(np.array(range(epochs_total)), history['accuracy'], label='accuracy')
  plt.plot(np.array(range(epochs_total)), history['test loss'], label='test loss')
  plt.plot(np.array(range(epochs_total)), history['test accuracy'], label='test accuracy')
  plt.legend()
  plt.savefig(modelName + '_train_loss_acc.png')
  plt.clf()
  """

  torch.save(model, "trained_models/" + modelName + ".pt")

"""# Evaluation"""

trainedModelPath = "./trained_models/" + modelName + ".pt"
model_loaded = torch.load(trainedModelPath, map_location='cpu')
model = classifier(full=model_loaded.full, binary=model_loaded.binary).to(device)
model.load_state_dict(model_loaded.state_dict())

"""## Evaluation functions"""

"""## Evaluation"""

"""
device_eval = 'cpu'

trainedModelPath = "./trained_models/" + modelName + ".pt"
model = torch.load(trainedModelPath, map_location=device_eval)
#quantized_model.load_state_dict(models[0][1].state_dict())
#quantized_model = models[0][1]
#quantizeModel(quantized_model)

models_eval = [(modelName, model)]

for (model_name, model) in models_eval:
  
  test_loss_meter = AverageMeter()
  test_accuracy_meter = AverageMeter()

  model.eval()
  eval_dataset_loader = test_dataset_loader
  for images, labels in eval_dataset_loader:
    images = images.to(device_eval)
    labels = labels.to(device_eval)

    no_images = images.shape[0]
    with torch.no_grad():
      labels_pred = model.forward(images)
    loss = loss_metric(labels_pred, labels)
    accuracy = torch.sum(torch.argmax(labels_pred, dim=1) == labels) / no_images

    test_loss_meter.update(loss, no_images)
    test_accuracy_meter.update(accuracy, no_images)

    print("\rTest loss: {0:.4f} Test accuracy: {1:.4f}".format(test_loss_meter.avg, test_accuracy_meter.avg), end='')
    log.write("\rTest loss: {0:.4f} Test accuracy: {1:.4f}\n".format(test_loss_meter.avg, test_accuracy_meter.avg))
    log.flush()
"""

save_dir = "./training_data/conv/"
test_start_id = 0
test_end_id = 100
size = 0
# Extract data
conv_count = 0
bn2d_count = 0
bn1d_count = 0
fc_count = 0

upload_dir = save_dir
creat_dir=True
for mod in model.modules():
    if isinstance(mod, nn.Conv2d):
        print(mod)
        weight = mod.weight.data.type(torch.int16).cpu().numpy()
        
        conv_count += 1
        if permute==1 and hasattr(mod,'permute_list'): 
            lis = mod.permute_list.cpu().numpy()
            os.chdir('..')
            for root, dirs, files in os.walk("."):
                for name in dirs:
                    if size == 0:
                        if fnmatch.fnmatch(name, "CNN_Medium.nnef"):
                            creat_dir = False
                    else:
                        if fnmatch.fnmatch(name, "CNN_Large.nnef"):
                            creat_dir = False
            if creat_dir:
                if size == 0:
                    os.mkdir("CNN_Medium.nnef")
                    os.chdir("CNN_Medium.nnef")
                else:
                    os.mkdir("CNN_Large.nnef")
                    os.chdir("CNN_Large.nnef")
            else:
                if size == 0:
                    os.chdir("CNN_Medium.nnef")
                else:
                    os.chdir("CNN_Large.nnef")
            np.save('conv_{0}_list'.format(conv_count), lis)   
            os.chdir('..')
            os.chdir('3pxnet-training')
        np.save(upload_dir+'conv_{0}_weight'.format(conv_count), weight)
    if isinstance(mod, nn.BatchNorm2d):
        print(mod)
        weight = mod.weight.data.cpu().numpy()
        bias = mod.bias.data.cpu().numpy()
        mean = mod.running_mean.cpu().numpy()
        var = mod.running_var.cpu().numpy()

        bn2d_count += 1
        np.save(upload_dir+'bn2d_{0}_weight'.format(bn2d_count), weight)
        np.save(upload_dir+'bn2d_{0}_bias'.format(bn2d_count), bias)
        np.save(upload_dir+'bn2d_{0}_mean'.format(bn2d_count), mean)
        np.save(upload_dir+'bn2d_{0}_var'.format(bn2d_count), var)

    if isinstance(mod, nn.Linear):
        print(mod)
        weight = mod.weight.data.type(torch.int16).cpu().numpy()
        fc_count += 1
        if permute==1 and hasattr(mod,'permute_list'): 
            lis = mod.permute_list.cpu().numpy()
            os.chdir('..')
            for root, dirs, files in os.walk("."):
                for name in dirs:
                    if size==0:
                        if fnmatch.fnmatch(name, "CNN_Medium.nnef"):
                            creat_dir=False
                    else:
                        if fnmatch.fnmatch(name, "CNN_Large.nnef"):
                            creat_dir=False
            if creat_dir:
                if size==0:
                    os.mkdir("CNN_Medium.nnef")
                    os.chdir("CNN_Medium.nnef")
                else:
                    os.mkdir("CNN_Large.nnef")
                    os.chdir("CNN_Large.nnef")
            else:
                if size==0:
                    os.chdir("CNN_Medium.nnef")
                else:
                    os.chdir("CNN_Large.nnef")
            np.save('fc_{0}_list'.format(fc_count), lis)
            os.chdir('..')
            os.chdir('3pxnet-training')
        np.save(upload_dir+'fc_{0}_weight'.format(fc_count), weight)
    if isinstance(mod, nn.BatchNorm1d):
        print(mod)
        bn1d_count += 1
        if type(mod.weight) != type(None):
            weight = mod.weight.data.cpu().numpy()
            np.save(upload_dir+'bn1d_{0}_weight'.format(bn1d_count), weight)
        if type(mod.bias) != type(None):
            bias = mod.bias.data.cpu().numpy()
            np.save(upload_dir+'bn1d_{0}_bias'.format(bn1d_count), bias)
        mean = mod.running_mean.cpu().numpy()
        var = mod.running_var.cpu().numpy()

        np.save(upload_dir+'bn1d_{0}_mean'.format(bn1d_count), mean)
        np.save(upload_dir+'bn1d_{0}_var'.format(bn1d_count), var)

x=Variable(torch.randn(1,3,32,32,requires_grad=True).to(device))
torch.onnx.export(model,x,"training_data/CNN_Medium.onnx",verbose=True,opset_version=9,input_names = ['input'], output_names = ['output'])
model_onnx=onnx.load("training_data/CNN_Medium.onnx")
# this can remove unecessary nodes
ort_session = onnxruntime.InferenceSession("training_data/CNN_Medium.onnx")

device = torch.device('cpu')
model.to(device)
testdata=torch.from_numpy(test_dataset_loader.dataset.data[test_start_id:test_end_id]).permute(0,3,1,2).float()
"""
os.chdir('..')
os.chdir('3pxnet-compiler')
temp = open('__Golden.txt', 'w+')
re = model(testdata)
re = re.tolist()
temp.write("Testing compiler output with golden output on CIFAR \n")
if permute == 1:
    temp.write("The network is in 3PXNet style \n")
else:
    if model.binary:
        temp.write("The network is dense and binary \n")
    else:
        temp.write("The network is pruned and packed \n")
if size == 0:
    temp.write("The network is a small CNN network \n")
else:
    temp.write("The network is a large CNN network \n")
for i in range(test_end_id-test_start_id):
    temp.write(str(re[i].index(max(re[i]))))
    temp.write(' ')
temp.close()
"""
log.close()
