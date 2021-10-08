import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import os
import math
from matplotlib import pyplot as plt

from utils import *
import utils_own
import network
from network import pnet, rnet, onet
from face_detection_dataset import MTCNNTrainDataset
import esp_dl_utils

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

torch.manual_seed(0)
np.random.seed(0)

EPOCHS = 25 # 25
EPOCHS_2 = 200 # 200

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
 
def forward(data_loader, model, score_criterion, box_landmark_criterion, epoch=0, training=True, optimizer=None, verbal=False): 
  logName = ""

  score_losses = AverageMeter()
  box_losses = AverageMeter()
  landmark_losses = AverageMeter()
  score_accs = AverageMeter()
  score_f1s = AverageMeter()

  if model.name not in ["pnet", "rnet", "onet"]:
    raise Exception("Invalid model name %s" % model.name)
  
  BS = 32
  dataset_loader = DataLoader(data_loader, batch_size=BS, shuffle=True)
  batch = 0
  batches_total = math.ceil(float(len(data_loader)) / BS)
  for image, score in dataset_loader:
      input_var = Variable(image).to(device)
      score_var = Variable(score).to(device)
      """
      if box is not None:
        box_var = Variable(box)
      if landmark is not None:
        landmark_var = Variable(landmark)
      """
      if model.name in ["pnet", "rnet"]:
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

  prefix = 'Training ' if training else 'Validation '
  if training:
    message = '\nEpoch: [{0}]\t'.format(epoch)
  else:
    message = ''
  message += prefix + 'score acc {score_acc.avg:.4f}\t'.format(score_acc=score_accs)
  message += prefix + 'score f1 {score_f1.avg:.4f}\t'.format(score_f1=score_f1s)
  message += prefix + 'score loss {score_loss.avg:.4f}\t'.format(score_loss=score_losses)
  message += prefix + 'box loss {box_loss.avg:.4f}\t'.format(box_loss=box_losses)
  if model.name == "onet":
    message += prefix + 'landmark loss {landmark_loss.avg:.4f}\t'.format(landmark_loss=landmark_losses)
  with open(os.path.join("training_logs", model.name, model.filename + ".txt"), "a") as f:
    f.write(message) 
  if epoch%10==9 and verbal:
    print(message, end='')
      
  return score_accs.avg, score_f1s.avg, score_losses.avg

def train(data_loader, model, score_criterion, box_landmark_criterion, epoch, optimizer, verbal=False):
   # switch to train mode
   model.train()
   return forward(data_loader, model, score_criterion, box_landmark_criterion, epoch, training=True, optimizer=optimizer, verbal=verbal)

def validate(data_loader, model, score_criterion, box_landmark_criterion, epoch, verbal=False):
   # switch to evaluate mode
   model.eval()
   return forward(data_loader, model, score_criterion, box_landmark_criterion, epoch, training=False, optimizer=None, verbal=verbal)

def train_all(models):
  pnet_trainloader, pnet_testloader, rnet_trainloader, rnet_testloader, onet_trainloader, onet_testloader = get_dataset()
  learning_rate = 1e-4
  score_criterion = nn.BCEWithLogitsLoss()
  box_landmark_criterion = nn.MSELoss()
  lr_decay = np.power((2e-6/learning_rate), (1./100))
  pack=32
  
  for model in models:
    print("Begin training", model.filename)

    trainHistory = {"train loss": [], "train acc": [], "train f1": [], "val loss": [], "val acc": [], "val f1": []}

    save_file = os.path.join(os.getcwd(), "torch_saved_models", model.name, model.filename + ".pt")

    if model.name == "pnet":
      trainloader, testloader = pnet_trainloader, pnet_testloader
    elif model.name == "rnet":
      trainloader, testloader = rnet_trainloader, rnet_testloader
    elif model.name == "onet":
      trainloader, testloader = onet_trainloader, onet_testloader

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_decay)

    utils_own.adjust_pack(model, 1)
    for epoch in range(0, EPOCHS):
      train_score_acc, train_score_f1, train_score_loss = train(trainloader, model, score_criterion, box_landmark_criterion, epoch, optimizer)
      val_score_acc, val_score_f1, val_score_loss = validate(testloader, model, score_criterion, box_landmark_criterion, epoch)
      scheduler.step()

      trainHistory["train loss"].append(train_score_loss)
      trainHistory["train acc"].append(train_score_acc)
      trainHistory["train f1"].append(train_score_f1)
      trainHistory["val loss"].append(val_score_loss)
      trainHistory["val acc"].append(val_score_acc)
      trainHistory["val f1"].append(val_score_f1)

    utils_own.adjust_pack(model, pack)
    utils_own.permute_all_weights_once(model, pack=pack, mode=-1)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_decay)

    for epoch in range(0, EPOCHS):
      train_score_acc, train_score_f1, train_score_loss = train(trainloader, model, score_criterion, box_landmark_criterion, epoch, optimizer)
      val_score_acc, val_score_f1, val_score_loss = validate(testloader, model, score_criterion, box_landmark_criterion, epoch)
      scheduler.step()

      trainHistory["train loss"].append(train_score_loss)
      trainHistory["train acc"].append(train_score_acc)
      trainHistory["train f1"].append(train_score_f1)
      trainHistory["val loss"].append(val_score_loss)
      trainHistory["val acc"].append(val_score_acc)
      trainHistory["val f1"].append(val_score_f1)

    # Fix pruned packs and fine tune
    for mod in model.modules():
      if hasattr(mod, 'mask'):
         mod.mask = torch.abs(mod.weight.data)
    model.pruned = True

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_decay)
    lowest_loss = 100 #starting value

    best_score_epoch = None
    best_score_acc = 0
    best_score_f1 = None
    best_score_loss = None
    for epoch in range(0, EPOCHS_2):
      train_score_acc, train_score_f1, train_score_loss = train(trainloader, model, score_criterion, box_landmark_criterion, epoch, optimizer)
      val_score_acc, val_score_f1, val_score_loss = validate(testloader, model, score_criterion, box_landmark_criterion, epoch)
      scheduler.step()

      trainHistory["train loss"].append(train_score_loss)
      trainHistory["train acc"].append(train_score_acc)
      trainHistory["train f1"].append(train_score_f1)
      trainHistory["val loss"].append(val_score_loss)
      trainHistory["val acc"].append(val_score_acc)
      trainHistory["val f1"].append(val_score_f1)

      # remember best loss and save checkpoint
      if val_score_acc > best_score_acc:
         best_score_epoch = epoch
         best_score_acc = val_score_acc
         best_score_f1 = val_score_f1
         best_score_loss = val_score_loss
         
         torch.save(model, save_file)

    trainHistory["best epoch"] = best_score_epoch

    message = 'Best Epoch: [{0}]\t'.format(best_score_epoch)
    message += 'val score acc {0:.4f}\t'.format(best_score_acc)
    message += 'val score f1 {0:.4f}\t'.format(best_score_f1)
    message += 'val score loss {0:.4f}'.format(best_score_loss)
    print(message)
    with open(os.path.join("training_logs", model.name, model.filename + ".txt"), "a") as f:
      f.write("\n" + message)

    #history[model.filename] = trainHistory
    filename, data = model.filename, trainHistory
    plt.clf()

    train_loss_line, = plt.plot(data["train loss"], label="train loss")
    train_acc_line, = plt.plot(data["train acc"], label="train acc")
    train_f1_line, = plt.plot(data["train f1"], label="train f1")
    val_loss_line, = plt.plot(data["val loss"], label="val loss")
    val_acc_line, = plt.plot(data["val acc"], label="val acc")
    val_f1_line, = plt.plot(data["val f1"], label="val f1")

    """
    train_loss_line.set_label("train loss")
    train_acc_line.set_label("train acc")
    train_f1_line.set_label("train f1")
    val_loss_line.set_label("val loss")
    val_acc_line.set_label("val acc")
    val_f1_line.set_label("val f1")
    """

    plt.axvline(x=EPOCHS+EPOCHS+data["best epoch"])

    plt.legend()
    plt.xlabel("Epoch")
    plt.title(filename + " training history")
    plt.savefig(os.path.join("training_history_imgs", filename.split("_")[0], filename + ".png"))

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
    for model in models:
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

      if "pnet" in model.name:
        dim = 12
      elif "rnet" in model.name:
        dim = 24
      elif "onet" in model.name:
        dim = 48
      else: #error
        dim = -1

      model.eval()
      onnxFilename = os.path.join("training_data", model.filename + ".onnx")
      size = (1, 3, dim, dim)
      x=Variable(torch.randn(size,requires_grad=True).to(device))
      #for binarized/ternarized networks, model.forward is necessary for weights to be appropriately binarized and sparsified
      with torch.no_grad():
        torch_pred = [out.cpu().detach().numpy().flatten() for out in model.forward(x)]
        print(model.name, "torch prediction", [out[:10] for out in torch_pred])
      torch.onnx.export(model,x,onnxFilename,opset_version=9,input_names = ['input'], output_names = ['output'])

if __name__ == "__main__":
  sparsity_low = 0.1
  sparsity_medium = 0.5
  sparsity_high = 0.9

  pnet_model_full = pnet(full=True).to(device)
  rnet_model_full = rnet(full=True).to(device)
  onet_model_full = onet(full=True).to(device)
  pnet_model_binarized = pnet(full=False, binary=True).to(device)
  rnet_model_binarized = rnet(full=False, binary=True).to(device)
  onet_model_binarized = onet(full=False, binary=True).to(device)
  pnet_model_2bit = pnet(full=False, binary=True, bitwidth=2).to(device)
  rnet_model_2bit = rnet(full=False, binary=True, bitwidth=2).to(device)
  onet_model_2bit = onet(full=False, binary=True, bitwidth=2).to(device)
  pnet_model_4bit = pnet(full=False, binary=True, bitwidth=4).to(device)
  rnet_model_4bit = rnet(full=False, binary=True, bitwidth=4).to(device)
  onet_model_4bit = onet(full=False, binary=True, bitwidth=4).to(device)
  pnet_model_ternarized_low = pnet(full=False, binary=False, conv_thres=sparsity_low).to(device)
  rnet_model_ternarized_low = rnet(full=False, binary=False, first_sparsity=sparsity_low, rest_sparsity=sparsity_low, conv_thres=sparsity_low).to(device)
  onet_model_ternarized_low = onet(full=False, binary=False, first_sparsity=sparsity_low, rest_sparsity=sparsity_low, conv_thres=sparsity_low).to(device)
  pnet_model_ternarized_medium = pnet(full=False, binary=False, conv_thres=sparsity_medium).to(device)
  rnet_model_ternarized_medium = rnet(full=False, binary=False, first_sparsity=sparsity_medium, rest_sparsity=sparsity_medium, conv_thres=sparsity_medium).to(device)
  onet_model_ternarized_medium = onet(full=False, binary=False, first_sparsity=sparsity_medium, rest_sparsity=sparsity_medium, conv_thres=sparsity_medium).to(device)
  pnet_model_ternarized_high = pnet(full=False, binary=False, conv_thres=sparsity_high).to(device)
  rnet_model_ternarized_high = rnet(full=False, binary=False, first_sparsity=sparsity_high, rest_sparsity=sparsity_high, conv_thres=sparsity_high).to(device)
  onet_model_ternarized_high = onet(full=False, binary=False, first_sparsity=sparsity_high, rest_sparsity=sparsity_high, conv_thres=sparsity_high).to(device)
  
  onet_model_binarized_inputbw1 = onet(full=False, binary=True, input_bitwidth=1).to(device)
  onet_model_binarized_inputbw2 = onet(full=False, binary=True, input_bitwidth=2).to(device)
  onet_model_binarized_inputbw3 = onet(full=False, binary=True, input_bitwidth=3).to(device)
  onet_model_binarized_inputbw4 = onet(full=False, binary=True, input_bitwidth=4).to(device)

  full_models = [pnet_model_full, rnet_model_full, onet_model_full]
  _4bit_models = [pnet_model_4bit, rnet_model_4bit, onet_model_4bit]
  _2bit_models = [pnet_model_2bit, rnet_model_2bit, onet_model_2bit]
  binarized_models = [pnet_model_binarized, rnet_model_binarized, onet_model_binarized]
  ternarized_low_models = [pnet_model_ternarized_low, rnet_model_ternarized_low, onet_model_ternarized_low]
  ternarized_medium_models = [pnet_model_ternarized_medium, rnet_model_ternarized_medium, onet_model_ternarized_medium]
  ternarized_high_models = [pnet_model_ternarized_high, rnet_model_ternarized_high, onet_model_ternarized_high]

  pnet_models = [
    pnet_model_full,
    pnet_model_binarized,
    pnet_model_ternarized_low,
    pnet_model_ternarized_medium,
    pnet_model_ternarized_high,
    pnet_model_2bit, 
    pnet_model_4bit, 
  ]
  rnet_models = [
    rnet_model_full,
    rnet_model_binarized,
    rnet_model_ternarized_low,
    rnet_model_ternarized_medium,
    rnet_model_ternarized_high,
    rnet_model_2bit, 
    rnet_model_4bit, 
  ]
  onet_models = [
    onet_model_full,
    onet_model_binarized,
    onet_model_ternarized_low,
    onet_model_ternarized_medium,
    onet_model_ternarized_high,
    onet_model_2bit, 
    onet_model_4bit, 
  ]
  onet_input_bitwidth_models = [
    onet_model_binarized_inputbw1,
    onet_model_binarized_inputbw2,
    onet_model_binarized_inputbw3,
    onet_model_binarized_inputbw4,
  ]


  for models in [onet_input_bitwidth_models]:
    train_all(models)
    #save_onnx(models)
  #write_esp_dl_headers(full_models[0][1], full_models[1][1], full_models[2][1])
  
