import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import os

from utils import *
import network
import utils_own
from face_detection_dataset import MTCNNTrainDataset
from network import pnet, rnet, onet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(0)
np.random.seed(0)

def calcAccuracy(pred_score, actual_score):
  #acc = torch.mean((torch.argmax(pred_score, dim=1) == actual_score).type(torch.FloatTensor))
  pred = torch.argmax(pred_score, dim=1).cpu().numpy().flatten()
  true = actual_score.cpu().numpy().flatten()
  acc = accuracy_score(true, pred)
  f1 = f1_score(true, pred)
  return acc, f1

def forward(data_loader, model, modelName, score_criterion, box_landmark_criterion, epoch=0, training=True, optimizer=None, verbal=False, keras=False):
  if keras:
    assert not training, "keras option not compatible with training mode"
  
  score_losses = AverageMeter()
  box_losses = AverageMeter()
  landmark_losses = AverageMeter()
  score_accs = AverageMeter()
  score_f1s = AverageMeter()

  if modelName not in ["pnet", "rnet", "onet"]:
    raise Exception("Invalid model name %s" % modelName)
  
  data_loader.change_model(modelName)

  data_loader_iter = iter(data_loader)
  batch = 0
  batches_total = data_loader.get_total_batches()
  if training:
    print()
  for choice, image, (box, landmark, score) in data_loader_iter:
      if training and batch % 100 == 0:
        message = '\rEpoch: [{0}]\t'.format(epoch)
        message += 'Batch {0}/{1}\t'.format(batch, batches_total)
        message += 'Training score acc {score_acc.avg:.4f}\t'.format(score_acc=score_accs)
        message += 'Training score f1 {score_f1.avg:.4f}\t'.format(score_f1=score_f1s)
        message += 'Training score loss {score_loss.avg:.4f}\t'.format(score_loss=score_losses)
        message += 'Training box loss {box_loss.avg:.4f}\t'.format(box_loss=box_losses)
        if modelName == 'onet':
          message += 'Training landmark loss {landmark_loss.avg:.4f}\t'.format(landmark_loss=landmark_losses)
        print(message, end='')

      input_var = Variable(image)
      score_var = Variable(score)
      if box is not None:
        box_var = Variable(box)
      if modelName in ["pnet", "rnet"]:
        score_out, box_out = model(input_var)
      else:
        score_out, box_out, landmark_out = model(input_var)
      if landmark is not None:
        landmark_var = Variable(landmark)

      score_var = torch.tensor(score_var)
      score_loss = score_criterion(score_out, score_var)
      score_losses.update(score_loss.item(), image.size(0))

      score_out = torch.flatten(score_out, start_dim=1)
      score_var = torch.flatten(score_var, start_dim=1)
      score_acc, score_f1 = calcAccuracy(score_out, torch.argmax(score_var, dim=1))
      score_accs.update(score_acc, image.size(0))
      score_f1s.update(score_f1, image.size(0))

      loss = score_loss
      if choice == 0:
        box_loss = box_landmark_criterion(box_out, box_var)*1e3
        box_losses.update(box_loss.item(), image.size(0))
        loss += box_loss
      elif choice == 1:
        landmark_loss = box_landmark_criterion(landmark_out, landmark_var)*1e3
        landmark_losses.update(landmark_loss.item(), image.size(0))
        loss += landmark_loss
      score_acc = calcAccuracy(score_out, torch.argmax(score_var, dim=1))

      if training:
         # compute gradient and do SGD step
         optimizer.zero_grad()
         loss.backward()
         for p in model.modules():
            if hasattr(p, 'weight_org'):
               p.weight.data.copy_(p.weight_org)
         optimizer.step()
         for p in model.modules():
            if hasattr(p, 'weight_org'):
               p.weight_org.copy_(p.weight.data.clamp_(-1,1))
      batch += 1

  if verbal:
    if training:
      print('\r', end='')
    prefix = 'Training ' if training else 'Validation '
    if training:
      message = 'Epoch: [{0}]\t'.format(epoch)
    else:
      message = ''
    message += prefix + 'score acc {score_acc.avg:.4f}\t'.format(score_acc=score_accs)
    message += prefix + 'score f1 {score_f1.avg:.4f}\t'.format(score_f1=score_f1s)
    message += prefix + 'score loss {score_loss.avg:.4f}\t'.format(score_loss=score_losses)
    message += prefix + 'box loss {box_loss.avg:.4f}\t'.format(box_loss=box_losses)
    if modelName == "onet":
      message += prefix + 'landmark loss {landmark_loss.avg:.4f}\t'.format(landmark_loss=landmark_losses)
    print(message, end='')
      
  return score_losses.avg, box_losses.avg

def train(data_loader, model, modelName, score_criterion, box_landmark_criterion, epoch, optimizer, verbal=True):
   # switch to train mode
   model.train()
   return forward(data_loader, model, modelName, score_criterion, box_landmark_criterion, epoch, training=True, optimizer=optimizer, verbal=verbal)

def validate(data_loader, model, modelName, score_criterion, box_landmark_criterion, epoch, verbal=True):
   # switch to evaluate mode
   model.eval()
   return forward(data_loader, model, modelName, score_criterion, box_landmark_criterion, epoch, training=False, optimizer=None, verbal=verbal)

suffix = "_small"

root = os.path.join(os.getcwd(), "training_data", "revisedDataset")
box_dataset = np.load(os.path.join(root, 'box_dataset' + suffix + '.npz'))
negative_dataset = np.load(os.path.join(root, 'negative_dataset' + suffix + '.npz'))
landmark_dataset = np.load(os.path.join(root, 'landmark_dataset' + suffix + '.npz'))

print("Reading box dataset")
box_images = box_dataset['images']
box_boxes = box_dataset['boxes']
print("Finished reading box dataset")
print("Reading negative dataset")
negative_images = negative_dataset['images']
print("Finished reading negative dataset")
print("Reading landmark dataset")
landmark_images = landmark_dataset['images']
landmark_landmarks = landmark_dataset['landmarks']
print("Finished reading landmark dataset")

data = {
  "box_images": box_images,
  "box_boxes": box_boxes,
  "negative_images": negative_images,
  "landmark_images": landmark_images,
  "landmark_landmarks": landmark_landmarks
}

choices = {
    0: "boxes",
    1: "landmarks",
    2: "negative"
}

batch=32

learning_rate = 1e-4
score_criterion = nn.BCEWithLogitsLoss()
box_landmark_criterion = nn.MSELoss()
lr_decay = np.power((2e-6/learning_rate), (1./100))

trainloader = MTCNNTrainDataset(root, data, batch_size=batch, train=True)
testloader = MTCNNTrainDataset(root, data, batch_size=batch, train=False)

full = True
binary = True
sparsity = 0.1

pnet_model = pnet(full=full, binary=binary, conv_thres=sparsity).to(device)
rnet_model = rnet(full=full, binary=binary, first_sparsity=sparsity, rest_sparsity=sparsity, conv_thres=sparsity).to(device)
onet_model = onet(full=full, binary=binary, first_sparsity=sparsity, rest_sparsity=sparsity, conv_thres=sparsity).to(device)

models = [("pnet", pnet_model), ("rnet", rnet_model), ("onet", onet_model)]

for (modelName, model) in models:
  print("Begin training", modelName, end='')
  if model.full:
    extension = "_full"
  else:
    if model.binary:
      extension = "_binary"
    else:
      extension = "_ternary"
  save_file = modelName + "_model" + extension + ".pt"

  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_decay)

  EPOCHS = 50
  for epoch in range(0, EPOCHS):
    train_score_loss, train_box_loss = train(trainloader, model, modelName, score_criterion, box_landmark_criterion, epoch, optimizer, verbal=True)
    val_score_loss, val_box_loss = validate(testloader, model, modelName, score_criterion, box_landmark_criterion, epoch, verbal=True)
    scheduler.step()

  torch.save(model, save_file)

  utils_own.adjust_pack(net, 1)
  utils_own.permute_all_weights_once(net, pack=pack, mode=-1)

  optimizer = optim.Adam(net.parameters(), lr=learning_rate)
  scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_decay)

  for epoch in range(0, EPOCHS):
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

  for epoch in range(0, EPOCHS):
    train_score_loss, train_box_loss = train(trainloader, net, score_criterion, box_landmark_criterion, epoch, optimizer)
    val_score_loss, val_box_loss = validate(testloader, net, score_criterion, box_landmark_criterion, epoch, verbal=True)

    # remember best loss and save checkpoint
    is_best = val_score_loss + val_box_loss < lowest_loss
    if is_best:
       lowest_loss = val_score_loss + val_box_loss
       torch.save(net, save_file)
    scheduler.step()
