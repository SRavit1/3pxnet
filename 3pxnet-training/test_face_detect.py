import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import math
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import os

from utils import *
import utils_own
import network
from face_detection_dataset import MTCNNTrainDataset
from network import pnet, rnet, onet
import esp_dl_utils

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

torch.manual_seed(0)
np.random.seed(0)

EPOCHS = 1 # 25
EPOCHS_2 = 2 # 200

def calcAccuracy(pred_score, actual_score):
  #acc = torch.mean((torch.argmax(pred_score, dim=1) == actual_score).type(torch.FloatTensor))
  pred = torch.argmax(pred_score, dim=1).cpu().numpy().flatten()
  true = actual_score.cpu().numpy().flatten()
  acc = accuracy_score(true, pred)
  f1 = f1_score(true, pred)
  return acc, f1

def get_dataset():
  suffix = "_small"

  root = os.path.join(os.getcwd(), "data", "revisedDataset")
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

  pnet_trainloader = MTCNNTrainDataset(root, data, batch_size=batch, train=True, model='pnet')
  pnet_testloader = MTCNNTrainDataset(root, data, batch_size=batch, train=False, model='pnet')
  rnet_trainloader = MTCNNTrainDataset(root, data, batch_size=batch, train=True, model='rnet')
  rnet_testloader = MTCNNTrainDataset(root, data, batch_size=batch, train=False, model='rnet')
  onet_trainloader = MTCNNTrainDataset(root, data, batch_size=batch, train=True, model='onet')
  onet_testloader = MTCNNTrainDataset(root, data, batch_size=batch, train=False, model='onet')

  return pnet_trainloader, pnet_testloader, rnet_trainloader, rnet_testloader, onet_trainloader, onet_testloader

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
  
  #data_loader.change_model(modelName)

  #data_loader_iter = iter(data_loader)
  BS = 32
  dataset_loader = DataLoader(data_loader, batch_size=BS, shuffle=True)
  batch = 0
  batches_total = math.ceil(float(len(data_loader)) / BS)
  if training:
    print()
  #for choice, image, (box, landmark, score) in data_loader_iter:
  for image, score in dataset_loader:
      """
      if training and verbal and batch % 100 == 0:
        message = '\rEpoch: [{0}]\t'.format(epoch)
        message += 'Batch {0}/{1}\t'.format(batch, batches_total)
        message += 'Training score acc {score_acc.avg:.4f}\t'.format(score_acc=score_accs)
        message += 'Training score f1 {score_f1.avg:.4f}\t'.format(score_f1=score_f1s)
        message += 'Training score loss {score_loss.avg:.4f}\t'.format(score_loss=score_losses)
        message += 'Training box loss {box_loss.avg:.4f}\t'.format(box_loss=box_losses)
        if modelName == 'onet':
          message += 'Training landmark loss {landmark_loss.avg:.4f}\t'.format(landmark_loss=landmark_losses)
        print(message, end='')
      """

      input_var = Variable(image).to(device)
      score_var = Variable(score).to(device)
      """
      if box is not None:
        box_var = Variable(box)
      if landmark is not None:
        landmark_var = Variable(landmark)
      """
      if modelName in ["pnet", "rnet"]:
        score_out, box_out = model(input_var)
      else:
        score_out, box_out, landmark_out = model(input_var)

      score_var = torch.tensor(score_var)
      score_loss = score_criterion(score_out, score_var)
      score_losses.update(score_loss.item(), image.size(0))

      score_out = torch.flatten(score_out, start_dim=1)
      score_var = torch.flatten(score_var, start_dim=1)
      score_acc, score_f1 = calcAccuracy(score_out, torch.argmax(score_var, dim=1))
      score_accs.update(score_acc, image.size(0))
      score_f1s.update(score_f1, image.size(0))

      loss = score_loss
      """
      if choice == 0:
        box_loss = box_landmark_criterion(box_out, box_var)*1e3
        box_losses.update(box_loss.item(), image.size(0))
        loss += box_loss
      elif choice == 1:
        landmark_loss = box_landmark_criterion(landmark_out, landmark_var)*1e3
        landmark_losses.update(landmark_loss.item(), image.size(0))
        loss += landmark_loss
      """
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

def test_all(models, trainloader, testloader):
  learning_rate = 1e-4
  score_criterion = nn.BCEWithLogitsLoss()
  box_landmark_criterion = nn.MSELoss()
  lr_decay = np.power((2e-6/learning_rate), (1./100))
  pack=32

  for (modelName, model) in models:
    print("Begin training", modelName, "id", model.id, end='')
    if model.full:
      extension = "_full"
    else:
      if model.binary:
        extension = "_binary"
      else:
        extension = "_ternary"
    extension += "_bw" + str(model.bitwidth)
    save_file = modelName + "_model" + extension + "_" + model.id + ".pt"

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_decay)

    utils_own.adjust_pack(model, 1)
    for epoch in range(0, EPOCHS):
      #train_score_loss, train_box_loss = train(trainloader, model, modelName, score_criterion, box_landmark_criterion, epoch, optimizer, verbal=True)
      val_score_loss, val_box_loss = validate(testloader, model, modelName, score_criterion, box_landmark_criterion, epoch, verbal=True)
      scheduler.step()


    utils_own.adjust_pack(model, pack)
    utils_own.permute_all_weights_once(model, pack=pack, mode=-1)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_decay)

    for epoch in range(0, EPOCHS):
      #train_score_loss, train_box_loss = train(trainloader, model, modelName, score_criterion, box_landmark_criterion, epoch, optimizer)
      val_score_loss, val_box_loss = validate(testloader, model, modelName, score_criterion, box_landmark_criterion, epoch, verbal=True)
      scheduler.step()

    # Fix pruned packs and fine tune
    for mod in model.modules():
      if hasattr(mod, 'mask'):
         mod.mask = torch.abs(mod.weight.data)
    model.pruned = True

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_decay)
    lowest_loss = 100 #starting value

    for epoch in range(0, EPOCHS_2):
      #train_score_loss, train_box_loss = train(trainloader, model, modelName, score_criterion, box_landmark_criterion, epoch, optimizer)
      val_score_loss, val_box_loss = validate(testloader, model, modelName, score_criterion, box_landmark_criterion, epoch, verbal=True)

      # remember best loss and save checkpoint
      is_best = val_score_loss + val_box_loss < lowest_loss
      if is_best:
         lowest_loss = val_score_loss + val_box_loss
         torch.save(model, save_file)
      scheduler.step()

def write_esp_dl_headers(pnet_model, rnet_model, onet_model):
    weight = {
      "pnet_conv2d_kernel1": {
          "type": "conv_kernel",
          "value": list(pnet_model.conv1.parameters())[0]
      },
      "pnet_conv2d_bias1": {
          "type": "conv_bias",
          "value": torch.zeros(int(list(pnet_model.conv1.parameters())[0].shape[0]))
      },
      "pnet_conv2d_kernel2": {
          "type": "conv_kernel",
          "value": list(pnet_model.conv2.parameters())[0]
      },
      "pnet_conv2d_bias2": {
          "type": "conv_bias",
          "value": torch.zeros(int(list(pnet_model.conv2.parameters())[0].shape[0]))
      },
      "pnet_conv2d_kernel3": {
          "type": "conv_kernel",
          "value": list(pnet_model.conv3.parameters())[0]
      },
      "pnet_conv2d_bias3": {
          "type": "conv_bias",
          "value": torch.zeros(int(list(pnet_model.conv3.parameters())[0].shape[0]))
      },
      "pnet_conv2d_kernel4": {
          "type": "conv_kernel",
          "value": list(pnet_model.conv4.parameters())[0]
      },
      "pnet_conv2d_bias4": {
          "type": "conv_bias",
          "value": torch.zeros(int(list(pnet_model.conv4.parameters())[0].shape[0]))
      },
      "pnet_conv2d_kernel5": {
          "type": "conv_kernel",
          "value": list(pnet_model.conv5.parameters())[0]
      },
      "pnet_conv2d_bias5": {
          "type": "conv_bias",
          "value": torch.zeros(int(list(pnet_model.conv5.parameters())[0].shape[0]))
      },
      "rnet_conv2d_kernel1": {
          "type": "conv_kernel",
          "value": list(rnet_model.conv1.parameters())[0]
      },
      "rnet_conv2d_bias1": {
          "type": "conv_bias",
          "value": torch.zeros(int(list(rnet_model.conv1.parameters())[0].shape[0]))
      },
      "rnet_conv2d_kernel2": {
          "type": "conv_kernel",
          "value": list(rnet_model.conv2.parameters())[0]
      },
      "rnet_conv2d_bias2": {
          "type": "conv_bias",
          "value": torch.zeros(int(list(rnet_model.conv2.parameters())[0].shape[0]))
      },
      "rnet_conv2d_kernel3": {
          "type": "conv_kernel",
          "value": list(rnet_model.conv3.parameters())[0]
      },
      "rnet_conv2d_bias3": {
          "type": "conv_bias",
          "value": torch.zeros(int(list(rnet_model.conv3.parameters())[0].shape[0]))
      },
      "rnet_dense_kernel1": {
          "type": "dense_kernel",
          "value": list(rnet_model.fc1.parameters())[0]
      },
      "rnet_dense_kernel2": {
          "type": "dense_kernel",
          "value": list(rnet_model.fc2.parameters())[0]
      },
      "rnet_dense_kernel3": {
          "type": "dense_kernel",
          "value": list(rnet_model.fc3.parameters())[0]
      },
      "onet_conv2d_kernel1": {
          "type": "conv_kernel",
          "value": list(onet_model.conv1.parameters())[0]
      },
      "onet_conv2d_bias1": {
          "type": "conv_bias",
          "value": torch.zeros(int(list(onet_model.conv1.parameters())[0].shape[0]))
      },
      "onet_conv2d_kernel2": {
          "type": "conv_kernel",
          "value": list(onet_model.conv2.parameters())[0]
      },
      "onet_conv2d_bias2": {
          "type": "conv_bias",
          "value": torch.zeros(int(list(onet_model.conv2.parameters())[0].shape[0]))
      },
      "onet_conv2d_kernel3": {
          "type": "conv_kernel",
          "value": list(onet_model.conv3.parameters())[0]
      },
      "onet_conv2d_bias3": {
          "type": "conv_bias",
          "value": torch.zeros(int(list(onet_model.conv3.parameters())[0].shape[0]))
      },
      "onet_conv2d_kernel4": {
          "type": "conv_kernel",
          "value": list(onet_model.conv4.parameters())[0]
      },
      "onet_conv2d_bias4": {
          "type": "conv_bias",
          "value": torch.zeros(int(list(onet_model.conv4.parameters())[0].shape[0]))
      },
      "onet_dense_kernel1": {
          "type": "dense_kernel",
          "value": list(onet_model.fc1.parameters())[0]
      },
      "onet_dense_kernel2": {
          "type": "dense_kernel",
          "value": list(onet_model.fc2.parameters())[0]
      },
      "onet_dense_kernel3": {
          "type": "dense_kernel",
          "value": list(onet_model.fc3.parameters())[0]
      },
      "onet_dense_kernel4": {
          "type": "dense_kernel",
          "value": list(onet_model.fc4.parameters())[0]
      },
    }

    esp_dl_utils.writeFullPrecWeights(weight)
    esp_dl_utils.writeQuantizedWeights(weight)

def save_onnx(models):
    for (modelName, model) in models:
      if model.full:
        suffix = "_full"
      elif model.binary:
        suffix = "_binarized"
      else:
        suffix = "_ternarized"
        if model.conv_thres < 0.3:
          suffix += "_low"
        elif model.conv_thres < 0.7:
          suffix += "_medium"
        else:
          suffix += "_high"
      suffix += "_bw" + str(model.bitwidth)

      if "pnet" in modelName:
        dim = 12
      elif "rnet" in modelName:
        dim = 24
      elif "onet" in modelName:
        dim = 48
      else: #error
        dim = -1

      model.eval()
      onnxFilename = "training_data/" + modelName + suffix + "_" + model.id + ".onnx"
      size = (1, 3, dim, dim)
      x=Variable(torch.randn(size,requires_grad=True).to(device))
      #for binarized/ternarized networks, model.forward is necessary for weights to be appropriately binarized and sparsified
      with torch.no_grad():
        torch_pred = [out.cpu().detach().numpy().flatten() for out in model.forward(x)]
        print(modelName, "torch prediction", [out[:10] for out in torch_pred])
      torch.onnx.export(model,x,onnxFilename,opset_version=9,input_names = ['input'], output_names = ['output'])

def convertToQuantized(model):
  for param in model.parameters():
    quantized_value = torch.tensor(esp_dl_utils.getApproximateQuantizedWeight(param.detach())[1].numpy())
    with torch.no_grad():
      param.copy_(quantized_value)
    #for i in range(len(param.data)):
      #param.data[i] = esp_dl_utils.getApproximateQuantizedWeight(param.data[i])[1]
  return model

if __name__ == "__main__":
  pnet_model_full = torch.load("torch_saved_models/pnet/pnet_model_full_bw1_37b51e17.pt").to(device)
  rnet_model_full = torch.load("torch_saved_models/rnet/rnet_model_full_bw1_d0c9dcf7.pt").to(device)
  onet_model_full = torch.load("torch_saved_models/onet/onet_model_full_bw1_736d3f51.pt").to(device)

  pnet_model_qu = torch.load("torch_saved_models/pnet/pnet_model_full_bw1_37b51e17.pt").to(device)
  rnet_model_qu = torch.load("torch_saved_models/rnet/rnet_model_full_bw1_d0c9dcf7.pt").to(device)
  onet_model_qu = torch.load("torch_saved_models/onet/onet_model_full_bw1_736d3f51.pt").to(device)

  pnet_model_qu = convertToQuantized(pnet_model_qu)
  rnet_model_qu = convertToQuantized(rnet_model_qu)
  onet_model_qu = convertToQuantized(onet_model_qu)

  #pnet_model_binarized = torch.load("torch_saved_models/pnet/").to(device)
  #rnet_model_binarized = torch.load("torch_saved_models/rnet/").to(device)
  onet_model_binarized = torch.load("torch_saved_models/onet/onet_model_binary_bw1_be54da04.pt").to(device)

  models = [
    ("pnet", pnet_model_full),
    ("rnet", rnet_model_full),
    ("onet", onet_model_full),
    ("pnet", pnet_model_qu),
    ("rnet", rnet_model_qu),
    ("onet", onet_model_qu),
    ("onet", onet_model_binarized)
  ]
  models = [models[-1]]

  pnet_trainloader, pnet_testloader, rnet_trainloader, rnet_testloader,onet_trainloader, onet_testloader, = get_dataset()
  for model in [models]:
    """
    if model[0] == "pnet":
      trainloader, testloader = pnet_trainloader, pnet_testloader
    elif model[0] == "rnet":
      trainloader, testloader = rnet_trainloader, rnet_testloader
    elif model[0] == "onet":
      trainloader, testloader = onet_trainloader, onet_testloader
    test_all(model, trainloader, testloader)
    """
    save_onnx(models)
  #write_esp_dl_headers(full_models[0][1], full_models[1][1], full_models[2][1])
