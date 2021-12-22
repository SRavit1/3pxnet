import torch
import torch.nn as nn
import binarized_modules_multi

#conv_layer = binarized_modules_multi.BinarizeConv2d(1, 1, 32, 32, kernel_size=2, stride=1, padding=(0, 0), bias=False)

class Model(torch.nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.output_bw = 2
    self.conv_layer = torch.nn.Conv2d(32, 32, kernel_size=2, stride=1, padding=(0, 0), bias=False)
    #self.conv_layer.weight = torch.nn.Parameter(torch.ones(32, 32, 2, 2))
    self.conv_layer.weight = torch.nn.Parameter(torch.full((32, 32, 2, 2), 1.))

    self.bn1 = nn.BatchNorm2d(32)

  def forward(self, x):
    #return self.bn1(self.conv_layer.forward(x))
    result = self.conv_layer.forward(x)
    result = binarized_modules_multi.Binarize(result, quant_mode='multi', bitwidth=self.output_bw)
    print("input values (cwhn):", end=' ')
    print_array(x, 32)
    print("weight values (cwhn):", end=' ')
    print_array(self.conv_layer.weight, 32)
    print("output values (cwhn):", end=' ')
    print_array(result, 32)
    return result
    
def print_array(tensor, num):
  for elem in tensor.flatten()[:num]:
    print(elem.item(), end=", ")
  print()

if __name__ == "__main__":
  conv_model = Model()
  #x = torch.ones(1, 32, 5, 5)
  x = torch.full((1, 32, 5, 5), -1.)
  output = conv_model.forward(x)
  #print("output values (cwhn):", [int(elem) for elem in output.flatten()[:32]])

  conv_parameters = list(conv_model.conv_layer.parameters())[0]
  print("parameter count of nonnegative", str(int(torch.count_nonzero(torch.greater_equal(output, 0)))) + '/' + str(len(torch.flatten(output))))

  #torch.save(conv_model, "./validation_test.pt")
