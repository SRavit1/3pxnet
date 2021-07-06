from network import pnet
import utils_own

import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
from utils import *

def calcAccuracy(pred_score, actual_score):
	acc = torch.mean((torch.argmax(pred_score, dim=1) == actual_score).type(torch.FloatTensor))
	return acc

def forward(data_loader, model, score_criterion, box_landmark_criterion, epoch=0, training=True, optimizer=None, verbal=False):
   score_losses = AverageMeter()
   box_losses = AverageMeter()
   score_accs = AverageMeter()
   #top1 = AverageMeter()
   #top5 = AverageMeter()

   for i, (inputs, (score, box, landmark)) in enumerate(data_loader):
      # measure data loading time.

      score = score.to(model.conv1.weight.device)
      box = box.to(model.conv1.weight.device)
      landmark = landmark.to(model.conv1.weight.device)
      
      input_var = Variable(inputs.to(model.conv1.weight.device))
      score_var = Variable(score).type(torch.DoubleTensor)
      box_var = Variable(box)

      # compute output
      score_out, box_out = model(input_var)

      score_acc = calcAccuracy(score_out, torch.argmax(score_var, dim=1))

      score_loss = score_criterion(score_out, score_var)
      box_loss = box_landmark_criterion(box_out, box_var)*1e3
      loss = score_loss + box_loss

      # measure accuracy and record loss
      score_losses.update(score_loss.item(), inputs.size(0))
      box_losses.update(box_loss.item(), inputs.size(0))
      score_accs.update(score_acc, inputs.size(0))
      #prec1, prec5 = accuracy(score_out.data, score, topk=(1, 5))
      #top1.update(prec1.item(), inputs.size(0))
      #top5.update(prec5.item(), inputs.size(0))

      if training:
         # compute gradient and do SGD step
         optimizer.zero_grad()
         loss.backward()
         #box_loss.backward()
         for p in model.modules():
            if hasattr(p, 'weight_org'):
               p.weight.data.copy_(p.weight_org)
         optimizer.step()
         for p in model.modules():
            if hasattr(p, 'weight_org'):
               p.weight_org.copy_(p.weight.data.clamp_(-1,1))
   if not training:
      if verbal:
         print('Epoch: [{0}]\t'
               'Score loss {score_loss.val:.4f} ({score_loss.avg:.4f})\t'
               'Box loss {box_loss.val:.4f} ({box_loss.avg:.4f})\t'
               'Score acc {score_acc.val:.4f} ({score_acc.avg:.4f})\t'.format(
            epoch, score_loss=score_losses, box_loss=box_losses, score_acc=score_accs))

   return score_losses.avg, box_losses.avg

def train(data_loader, model, score_criterion, box_landmark_criterion, epoch, optimizer):
   # switch to train mode
   model.train()
   return forward(data_loader, model, score_criterion, box_landmark_criterion, epoch, training=True, optimizer=optimizer)


def validate(data_loader, model, score_criterion, box_landmark_criterion, epoch, verbal=False):
   # switch to evaluate mode
   model.eval()
   return forward(data_loader, model, score_criterion, box_landmark_criterion, epoch, training=False, optimizer=None, verbal=verbal)

full = True
binary = True
trainset, testset, classes = utils_own.load_dataset('CELEBA')
net = pnet(full=full, binary=binary)

batch=32
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch, shuffle=False, num_workers=2)

learning_rate = 1e-3
score_criterion = nn.BCEWithLogitsLoss()
box_landmark_criterion = nn.MSELoss()
lr_decay = np.power((2e-6/learning_rate), (1./100))

# Train without packing constraints
pack = 32

if full:
	extension = "_full"
else:
	if binary:
		extension = "_binary"
	else:
		extension = "_ternary"
save_file = "p_net_model"+extension+".pt"

utils_own.adjust_pack(net, 1)

optimizer = optim.Adam(net.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_decay)

epochs = 25
for epoch in range(0, epochs):
  train_score_loss, train_box_loss = train(trainloader, net, score_criterion, box_landmark_criterion, epoch, optimizer)
  val_score_loss, val_box_loss = validate(testloader, net, score_criterion, box_landmark_criterion, epoch, verbal=True)
  scheduler.step()

utils_own.adjust_pack(net, 1)
utils_own.permute_all_weights_once(net, pack=pack, mode=-1)

optimizer = optim.Adam(net.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_decay)

for epoch in range(0, epochs):
  train_score_loss, train_box_loss = train(trainloader, net, score_criterion, box_landmark_criterion, epoch, optimizer)
  val_score_loss, val_box_loss = validate(testloader, net, score_criterion, box_landmark_criterion, epoch, verbal=True)
  scheduler.step()

# Fix pruned packs and fine tune
for mod in net.modules():
  if hasattr(mod, 'mask'):
     mod.mask = torch.abs(mod.weight.data)
net.pruned = True

optimizer = optim.Adam(net.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_decay)
lowest_loss = 100 #starting value

for epoch in range(0, epochs):
  train_score_loss, train_box_loss = train(trainloader, net, score_criterion, box_landmark_criterion, epoch, optimizer)
  val_score_loss, val_box_loss = validate(testloader, net, score_criterion, box_landmark_criterion, epoch, verbal=True)

  # remember best prec@1 and save checkpoint
  is_best = val_score_loss + val_box_loss < lowest_loss
  if is_best:
     lowest_loss = val_score_loss + val_box_loss
     torch.save(net, save_file)
  scheduler.step()