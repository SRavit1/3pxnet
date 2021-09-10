import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os 
import numpy as np
import sklearn.metrics

from network import DeepAutoEncoder
import utils_own

model_path = "./trained_models"

full_model = DeepAutoEncoder()
full_model.load_state_dict(torch.load(os.path.join(model_path, "dae_full.pt")))

batch_size = 32
dataset, test_dataset, classes = utils_own.load_dataset('ToyADMOS')
test_dataset_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

models = [("dae_binarized", full_model)]

for (modelName, model) in models:
  y = np.array([])
  y_scores = np.array([])

  for batch_no, (inputs, labels) in enumerate(test_dataset_loader): 
    inputs_pred = model.forward(inputs).detach().numpy()
    inputs = inputs.detach().numpy()
    labels = labels.detach().numpy()
    loss = np.linalg.norm(inputs_pred - inputs, axis=1)**2
    
    y = np.append(y, labels)
    y_scores = np.append(y_scores, loss)

  fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, y_scores)
  auc_score = sklearn.metrics.auc(fpr, tpr)
  print("auc_score", auc_score)
