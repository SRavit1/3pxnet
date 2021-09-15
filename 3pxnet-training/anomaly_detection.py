import os
import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

import binarized_modules_multi as binarized_modules
import esp_dl_utils
import utils
import utils_own
import anomaly_detection_common as com
import network
 
def load_dataset(batch_size=512):
    dataset, test_dataset, classes = utils_own.load_dataset('ToyADMOS')

    dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_dataset_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return dataset_loader, test_dataset_loader

def train_single(model, optimizer, dataset_loader, test_dataset_loader, loss_metric, EPOCHS, history):
  total_batches = len(dataset_loader)

  for epoch in range(EPOCHS):
    loss_meter = utils.AverageMeter()
    test_loss_meter = utils.AverageMeter()

    model.train()
    for batch_no, (inputs, labels) in enumerate(dataset_loader):
      print("\rEpoch {0} Batch: {1}/{2} Loss: {3:.4f}".format(epoch, batch_no, total_batches, loss_meter.avg), end="")
      log.write("\rEpoch {0} Batch: {1}/{2} Loss: {3:.4f}".format(epoch, batch_no, total_batches, loss_meter.avg))
      log.flush()
      
      inputs = inputs.to(device)

      no_inputs = inputs.shape[0]
      inputs_pred = model.forward(inputs)
      loss = loss_metric(inputs_pred, inputs)

      optimizer.zero_grad()
      loss.backward()
      for p in model.modules():
        if hasattr(p, 'weight_org'):
            p.weight.data.copy_(p.weight_org)
      optimizer.step()
      for p in model.modules():
        if hasattr(p, 'weight_org'):
            p.weight_org.copy_(p.weight.data.clamp_(-1,1))

      loss_meter.update(loss, no_inputs)

    y_scores = np.array([])
    y = np.array([])

    model.eval()
    for inputs, labels  in test_dataset_loader:
      inputs = inputs.to(device)
      no_inputs = inputs.shape[0]
      labels_np = labels.cpu().detach().numpy()

      with torch.no_grad():
        inputs_pred = model.forward(inputs)

      mse_np = np.linalg.norm((inputs_pred - inputs).cpu().detach().numpy(), axis=1)**2
      y_scores = np.append(y_scores, mse_np)
      y = np.append(y, labels_np)

      inputs_normal = inputs[labels_np==0]
      no_inputs_normal = inputs_normal.shape[0]
      with torch.no_grad():
        inputs_normal_pred = model.forward(inputs_normal)

      loss = loss_metric(inputs_normal_pred, inputs_normal)
      test_loss_meter.update(loss, no_inputs_normal)
 
    #fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, y_scores)
    #auc_score = sklearn.metrics.auc(fpr, tpr)
    auc_score = utils_own.auc_metric(y_scores, y)
    print("\rEpoch {0} Loss: {1:.4f} Test loss: {2:.4f} Test AUC: {3:.4f}".format(epoch, 
      loss_meter.avg, test_loss_meter.avg, auc_score))
    log.write("\rEpoch {0} Loss: {1:.4f} Test loss: {2:.4f}, Test AUC: {3:.4f}\n".format(epoch, 
      loss_meter.avg, test_loss_meter.avg, auc_score))
    log.flush()

    history['loss'].append(float(loss_meter.avg))
    history['test loss'].append(float(test_loss_meter.avg))

def train(models):
    EPOCHS = 25
    loss_metric = nn.MSELoss()

    dataset_loader, test_dataset_loader = load_dataset()

    for (modelName, model) in models:
      pack = 32
      permute = 1

      print("Begin train", modelName)
      log.write("Begin train " + str(modelName) + "\n")
      log.flush()

      learning_rate = 1e-4
      lr_decay = np.power((2e-6/learning_rate), (1./100))
      optimizer = optim.Adam(model.parameters(), lr=learning_rate)

      utils_own.adjust_pack(model, 1)

      history = {'loss': [], 'test loss': []}
      train_single(model, optimizer, dataset_loader, test_dataset_loader, loss_metric, EPOCHS, history)

      # Retrain with permutation + packing constraint
      utils_own.adjust_pack(model, pack)
      utils_own.permute_all_weights_once(model, pack=pack, mode=permute)

      train_single(model, optimizer, dataset_loader, test_dataset_loader, loss_metric, EPOCHS, history)

      # Fix pruned packs and fine tune
      for mod in model.modules():
          if hasattr(mod, 'mask'):
              mod.mask = torch.abs(mod.weight.data)    
      model.pruned = True

      #torch.save(model.state_dict(), "trained_models/" + modelName + ".pt")

      train_single(model, optimizer, dataset_loader, test_dataset_loader, loss_metric, 200, history)

      epochs_total = EPOCHS + EPOCHS + 200

def save_onnx(models):
    for (modelName, model) in models:
      model.eval()
      onnxFilename = "training_data/" + modelName + ".onnx"
      x=Variable(torch.randn(32,640,requires_grad=True).to(device))
      #for binarized/ternarized networks, model.forward is necessary for weights to be appropriately binarized and sparsified
      with torch.no_grad():
        torch_pred = model.forward(x).cpu().detach().numpy().flatten()
        print(modelName, "torch prediction", torch_pred[:10])
      torch.onnx.export(model,x,onnxFilename,opset_version=9,input_names = ['input'], output_names = ['output'])

def write_esp_dl_headers(models):
    for (modelName, model) in models:
      if "full" not in modelName:
        print("Skipping non-full-precision model", modelName)
        continue
      
      weight = {
        "fc1_filter": {
            "type": "dense_kernel",
            "value": list(model.fc1.parameters())[0]
        },
        "fc2_filter": {
            "type": "dense_kernel",
            "value": list(model.fc2.parameters())[0]
        },
        "fc3_filter": {
            "type": "dense_kernel",
            "value": list(model.fc3.parameters())[0]
        },
        "fc4_filter": {
            "type": "dense_kernel",
            "value": list(model.fc4.parameters())[0]
        },
        "fc5_filter": {
            "type": "dense_kernel",
            "value": list(model.fc5.parameters())[0]
        },
        "fc6_filter": {
            "type": "dense_kernel",
            "value": list(model.fc6.parameters())[0]
        },
        "fc7_filter": {
            "type": "dense_kernel",
            "value": list(model.fc7.parameters())[0]
        },
        "fc8_filter": {
            "type": "dense_kernel",
            "value": list(model.fc8.parameters())[0]
        },
        "fc9_filter": {
            "type": "dense_kernel",
            "value": list(model.fc9.parameters())[0]
        },
        "fc10_filter": {
            "type": "dense_kernel",
            "value": list(model.fc10.parameters())[0]
        },
      }

      esp_dl_utils.writeFullPrecWeights(weight)
      esp_dl_utils.writeQuantizedWeights(weight)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log = open("log.txt", "w")

    print("DEVICE", device)
    log.write("DEVICE " + str(device) + "\n")

    classifier = network.DeepAutoEncoder

    full_model = classifier(True).to(device)
    binarized_model = classifier(False, True).to(device)
    ternarized_low_model = classifier(False, False, 0.1).to(device)
    ternarized_medium_model = classifier(False, False, 0.5).to(device)
    ternarized_high_model = classifier(False, False, 0.9).to(device)

    prefix = "dae_"
    models = [(prefix + "full", full_model), (prefix + "binarized", binarized_model), (prefix + "ternarized_low", ternarized_low_model), (prefix + "ternarized_medium", ternarized_medium_model), (prefix + "ternarized_high", ternarized_high_model)]

    train(models)
    save_onnx(models)

    log.close()
