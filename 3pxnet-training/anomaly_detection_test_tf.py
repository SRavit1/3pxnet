import tensorflow as tf
import numpy as np

#full_model = tf.keras.models.load_model("./trained_models/model_ToyCar.hdf5")
full_model = tf.keras.models.load_model("./trained_models/ad01.h5")
#print(full_model(np.random.randn(1, 640)))
models = [("full_model", full_model)]

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os 
import numpy as np
import sklearn.metrics

from network import DeepAutoEncoder
import utils_own

model_path = "./trained_models"


batch_size = 32
dataset, test_dataset, classes = utils_own.load_dataset('ToyADMOS')
test_dataset_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

models = [("dae_full", full_model)]

for (modelName, model) in models:
  y = np.array([])
  y_scores = np.array([])

  for batch_no, (inputs, labels) in enumerate(test_dataset_loader): 
    inputs_pred = model(inputs.detach().numpy())
    #inputs = inputs.detach().numpy()
    #labels = labels.detach().numpy()
    loss = np.linalg.norm(inputs_pred - inputs, axis=1)**2
    
    y = np.append(y, labels)
    y_scores = np.append(y_scores, loss)
    
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, loss)
    auc_score = sklearn.metrics.auc(fpr, tpr)
    print("\rBatch {0}/{1}. auc_score".format(batch_no, len(test_dataset_loader)), auc_score, end='')
    

  fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, y_scores)
  auc_score = sklearn.metrics.auc(fpr, tpr)
  print("\rauc_score", auc_score)
