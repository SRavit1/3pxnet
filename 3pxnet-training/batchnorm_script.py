import inspect
import torch
from bitarray import bitarray
import numpy as np
import torch.nn as nn
import math
def clamp(value, min, max):
   if (value < min):
      return min
   if (value > max):
      return max
   return value
#modelName = "./validation_test.pt"
modelName = "./torch_saved_models/onet/onet_model_binary_bw1_input_bw2_0a2c3f60.pt"
net=torch.load(modelName)
device = torch.device('cpu')
conv_count = 0
bn1d_count = 0
permute = 0
for i in inspect.getmembers(net):

   # to remove private and protected
   # functions
   if not i[0].startswith('_'):

      # To remove other methods that
      # doesnot start with a underscore
      if not inspect.ismethod(i[1]):
         print(i)
         mod = i[1]
         print(mod)
         if isinstance(mod, nn.Conv2d):
            weight = mod.weight.data.type(torch.int16).cpu().numpy()
            #et = mod.threshold.data.cpu().tolist()
            if i[0] == 'conv1':
               F = open("conv1_weight.h","w")
               F.write("#define _conv1_weight {\\\n")
               for out in range(mod.weight.shape[0]):
                  pack = bitarray()
                  for z in range(mod.weight.shape[1]):
                     for y in range(mod.weight.shape[2]):
                        for x in range(mod.weight.shape[3]):
                           if weight[out][z][y][x] > 0:
                              pack.append(1)
                           else:
                              pack.append(0)
                  F.write(str("0x%x" % int(pack.to01(), 2)) + ",")
                  F.write("\\\n")
               F.write("}")
            else:
               F = open(i[0]+"_weight.h","w")
               F.write("#define _"+i[0]+"_weight {\\\n")
               for out in range(mod.weight.shape[0]):
                  for y in range(mod.weight.shape[2]):
                     for x in range(mod.weight.shape[3]):
                        for z in range(int(mod.weight.shape[1]/32)):
                           pack = bitarray()
                           for temp in range(32):
                              if weight[out][z*32+temp][y][x] >= 0 :
                                 pack.append(1)
                              else:
                                 pack.append(0)
                           F.write(str("0x%x" % int(pack.to01(), 2)) + ", \\\n")
                     #F.write("\\\n")
               F.write("}\n")
               """F.write("#define _"+i[0]+"_threshold { ")
               for i in range(len(et)-1):
                  temp = str(et[i]).split(".")
                  b = abs(et[i] - int(temp[0]))
                  if et[i]<0:
                     temp[0] = str(bin(clamp(int(temp[0]), -32767, 32767)))[3:]
                  else:
                     temp[0] = str(bin(clamp(int(temp[0]), -32767, 32767)))[2:]
                  while len(temp[0]) <16:
                     temp[0] = "0" + temp[0]
                  temp[1] = str(bin(int(b * (2 ** 16))))[2:]
                  while len(temp[1]) < 16:
                     temp[1] = "0" + temp[1]
                  et_thres = temp[0] + temp[1]
                  if et[i] <0:
                     F.write("-")
                  F.write(str(int(et_thres,2))+", ")
               F.write("}")"""
            conv_count += 1

         if isinstance(mod, nn.BatchNorm2d) or isinstance(mod, nn.BatchNorm1d):
            weight = mod.weight.data.cpu().numpy()
            bias = mod.bias.data.cpu().numpy()
            mean = mod.running_mean.cpu().numpy()
            var = mod.running_var.cpu().numpy()
            F = open(i[0]+".h","w")
            F.write("#define "+i[0]+"_bias { ")
            for j in range(mod.num_features):
               F.write(str(mod.bias.data.cpu().tolist()[j]))
               F.write(", ")
            F.write("}\n")
            F.write("#define " + i[0] + "_weight { ")
            for j in range(mod.num_features):
               F.write(str(mod.weight.data.cpu().tolist()[j]))
               F.write(", ")
            F.write("}\n")
            F.write("#define " + i[0] + "_running_mean { ")
            for j in range(mod.num_features):
               F.write(str(mod.running_mean.data.cpu().tolist()[j]))
               F.write(", ")
            F.write("}\n")
            F.write("#define " + i[0] + "_running_var { ")
            for j in range(mod.num_features):
               F.write(str(math.sqrt(mod.running_var.data.cpu().tolist()[j])))
               F.write(", ")
            F.write("}\n")
            F.write("#define "+i[0]+"_thresh {\\\n ")
            for j in range(mod.num_features):
               #threshold = mean[j] - np.sqrt(var[j])/weight[j] * bias[j]
               old_threshold = mean[j] - np.sqrt(var[j])/weight[j] * bias[j]
               threshold = mean[j] - (bias[j]*var[j]/weight[j])
               threshold = old_threshold
               temp = str(threshold).split(".")
               b = abs(threshold - int(temp[0]))
               if threshold < 0:
                  temp[0] = str(bin(clamp(int(temp[0]), -32767, 32767)))[3:]
               else:
                  temp[0] = str(bin(clamp(int(temp[0]), -32767, 32767)))[2:]
               while len(temp[0]) < 16:
                  temp[0] = "0" + temp[0]
               temp[1] = str(bin(int(b * (2 ** 16))))[2:]
               while len(temp[1]) < 16:
                  temp[1] = "0" + temp[1]
               thres = temp[0] + temp[1]
               if threshold < 0:
                  F.write("-")
               F.write(str(int(thres, 2)) + ", \\\n")
               print("threshold, representation:", threshold, str(int(thres, 2)))
            F.write("} \n")

            sign = bitarray()
            F.write("#define "+i[0]+"_sign {\\\n")
            for pack in range(int(mod.num_features/32)):
               sign = bitarray()
               for temp in range(32):
                  sign.append(int(weight[pack*32+temp]>0))
               F.write(str("0x%x" % int(sign.to01(), 2))+ ", \\\n")
            F.write("} \n")

            F.write("#define "+i[0]+"_offset {\\\n")
            for j in range(mod.num_features):
               offset = np.sqrt(var[j])/weight[j]
               temp = str(offset).split(".")
               b = abs(offset - int(temp[0]))
               if offset < 0:
                  temp[0] = str(bin(clamp(int(temp[0]), -32767, 32767)))[3:]
               else:
                  temp[0] = str(bin(clamp(int(temp[0]), -32767, 32767)))[2:]
               while len(temp[0]) < 16:
                  temp[0] = "0" + temp[0]
               temp[1] = str(bin(int(b * (2 ** 16))))[2:]
               while len(temp[1]) < 16:
                  temp[1] = "0" + temp[1]
               thres = temp[0] + temp[1]
               if offset < 0:
                  F.write("-")
               F.write(str(int(thres, 2)) + ", \\\n")
            F.write("} \n")








         if isinstance(mod, nn.Linear):
            weight = mod.weight.data.type(torch.int16).cpu().numpy()
         if isinstance(mod, nn.BatchNorm1d):
            bn1d_count += 1
