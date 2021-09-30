import os
import torch
import numpy as np
import cv2
import math
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FaceLandmarksDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, train=False, model='onet'):
        self.root = root
        self.transform = transform

        dataset = np.load(os.path.join(root, 'CELEBA', 'small_dataset.npz'))

        all_image_data = dataset['all_image_data']
        self.all_box_data = dataset['all_box_data']
        self.all_landmark_data = dataset['all_landmark_data']
        self.all_score_data = dataset['all_score_data']
        #self.all_score_data = np.argmax(self.all_score_data, axis=1)

        self.model = model
        if model == 'pnet':
            self.all_image_data = np.empty((len(all_image_data), 12, 12, 3))
            for i in range(len(self.all_image_data)):
                self.all_image_data[i] = cv2.resize(all_image_data[i], (12, 12))
        elif model == 'rnet':
            self.all_image_data = np.empty((len(all_image_data), 24, 24, 3))
            for i in range(len(self.all_image_data)):
                self.all_image_data[i] = cv2.resize(all_image_data[i], (24, 24))
        elif model == 'onet':
            self.all_image_data = all_image_data
        else:
            raise Exception("Invalid model name %s" % model)

        cutoff = int(len(self.all_image_data)*0.9)
        if train:
            self.all_image_data = self.all_image_data[:cutoff]
            self.all_box_data = self.all_box_data[:cutoff]
            self.all_landmark_data = self.all_landmark_data[:cutoff]
            self.all_score_data = self.all_score_data[:cutoff]
        else:
            self.all_image_data = self.all_image_data[cutoff:]
            self.all_box_data = self.all_box_data[cutoff:]
            self.all_landmark_data = self.all_landmark_data[cutoff:]
            self.all_score_data = self.all_score_data[cutoff:]

        self.all_image_data = self.all_image_data.astype(np.float32)
        self.all_box_data = self.all_box_data.astype(np.float32)
        self.all_landmark_data = self.all_landmark_data.astype(np.float32)
        self.all_score_data = self.all_score_data.astype(np.float32)

        self.all_image_data /= 255
        self.len = len(self.all_image_data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.all_image_data[idx]
        box = self.all_box_data[idx]
        landmark = self.all_landmark_data[idx]
        score = self.all_score_data[idx]
        
        """
        sample = {'image': self.all_image_data[idx], 'box': self.all_box_data[idx], 
            'landmark':self.all_landmark_data[idx], 'score': self.all_score_data[idx]}
        """

        if self.transform:
            image = self.transform(image)

        return image, (score, box, landmark)

"""
class MTCNNTrainDataset(torch.utils.data.IterableDataset):
    def __init__(self, root, data, transform=None, train=True, model='pnet', batch_size=32):
        self.root = root
        self.transform = transform

        self.box_images = data["box_images"]
        self.box_boxes = data["box_boxes"]
        self.negative_images = data["negative_images"]
        self.landmark_images = data["landmark_images"]
        self.landmark_landmarks = data["landmark_landmarks"]

        cutoff_ratio = 0.9

        #train and test split
        box_cutoff = int(len(self.box_images)*cutoff_ratio)
        landmark_cutoff = int(len(self.landmark_images)*cutoff_ratio)
        negative_cutoff = int(len(self.negative_images)*cutoff_ratio)

        if train:
            self.box_images = self.box_images[:box_cutoff]
            self.box_boxes = self.box_boxes[:box_cutoff]
            self.landmark_images = self.landmark_images[:landmark_cutoff]
            self.landmark_landmarks = self.landmark_landmarks[:landmark_cutoff]
            self.negative_images = self.negative_images[:negative_cutoff]
        else:
            self.box_images = self.box_images[box_cutoff:]
            self.box_boxes = self.box_boxes[box_cutoff:]
            self.landmark_images = self.landmark_images[landmark_cutoff:]
            self.landmark_landmarks = self.landmark_landmarks[landmark_cutoff:]
            self.negative_images = self.negative_images[negative_cutoff:]

        # normalizing box and landmark coordinates between 0 and 1
        self.box_boxes = self.box_boxes/48
        self.landmark_landmarks = self.landmark_landmarks/48

        # subtracting one from last two values in box
        self.box_boxes[:,2] -= 1
        self.box_boxes[:,3] -= 1

        self.capacities = [len(self.box_images), len(self.landmark_images), len(self.negative_images)]
        self.change_batch_size(batch_size, reset=False)
        self.change_model(model, reset=False)
        self.reset_status()

    @staticmethod
    def preprocess_images(images, model):
      #resize according to model
      if model == 'pnet':
        dim = 12
      elif model == 'rnet':
        dim = 24
      elif model == 'onet':
        dim = 48
      else:
        raise Exception("Invalid model name %s" % model)
      
      images_resized = np.zeros((len(images), dim, dim, 3))
      for i in range(len(images)):
        images_resized[i] = cv2.resize(images[i], (dim, dim))

      #transposing nhwc to nchw
      images_resized = np.transpose(images_resized, (0, 3, 1, 2))

      #normalize between 0 and 1
      return images_resized/255
    
    def change_model(self, model, reset=True):
      if model == "pnet" or model == "rnet" or model == "onet":
        self.model = model
        if reset:
          self.reset_status()
      else:
        raise Exception("Invalid model name %s" % model)

      if model == 'pnet':
        self.box_boxes = np.reshape(self.box_boxes, (-1, 4, 1, 1))
      else:
        self.box_boxes = np.reshape(self.box_boxes, (-1, 4))
    
    def change_batch_size(self, batch_size: int, reset=True):
      assert batch_size > 0, "Invalid batch_size value %d." % batch_size
      self.batch_size = batch_size
      if reset:
        self.reset_status()
    
    def reset_status(self):
      self.array_indices = [0, 0, 0]
      if self.model == "onet":
        self.available_datasets = [0, 1, 2]
      else:
        self.available_datasets = [0, 2]
      self.len = sum([self.capacities[i] for i in self.array_indices])
      self.no_batches = math.ceil(self.len / self.batch_size)

      box_permutation = np.random.permutation(self.capacities[0])
      landmark_permutation = np.random.permutation(self.capacities[1])
      negative_permutation = np.random.permutation(self.capacities[2])

      self.box_images = self.box_images[box_permutation]
      self.box_boxes = self.box_boxes[box_permutation]
      self.landmark_images = self.landmark_images[landmark_permutation]
      self.landmark_landmarks = self.landmark_landmarks[landmark_permutation]
      self.negative_images = self.negative_images[negative_permutation]

    def get_total_batches(self):
      return self.no_batches

    def __len__(self):
        return self.len

    def __iter__(self):
        self.reset_status()
        return self

    def __next__(self):
        #return next value
        if len(self.available_datasets) == 0:
            raise StopIteration

        choice = random.choice(self.available_datasets)

        index = self.array_indices[choice]
        box = None
        landmark = None
        #box data
        if choice == 0:
            image = self.box_images[index:index+self.batch_size].copy()
            box = self.box_boxes[index:index+self.batch_size].copy()
            #score = np.array([[0, 1]]*len(image))
            score = np.array([[0.05, 0.95]]*len(image))
        #landmark data
        elif choice == 1:
            image = self.landmark_images[index:index+self.batch_size].copy()
            landmark = self.landmark_landmarks[index:index+self.batch_size].copy()
            #score = np.array([[0, 1]]*len(image))
            score = np.array([[0.05, 0.95]]*len(image))
        #negative data
        elif choice == 2:
            image = self.negative_images[index:index+self.batch_size].copy()
            #score = np.array([[1, 0]]*len(image))
            score = np.array([[0.95, 0.05]]*len(image))
        else:
            raise Exception("Invalid dataset choice", choice)

        if self.model == 'pnet':
          score = np.reshape(score, (-1, 2, 1, 1))

        self.array_indices[choice] += self.batch_size
        if self.array_indices[choice] >= self.capacities[choice]:
            self.available_datasets.remove(choice)

        if self.transform:
            image = self.transform(image)

        image = MTCNNTrainDataset.preprocess_images(image, self.model)

        image = torch.tensor(image.astype('float32')).to(device)
        if box is not None:
          box = torch.tensor(box.astype('float32')).to(device)
        if landmark is not None:
          landmark = torch.tensor(landmark.astype('float32')).to(device)
        score = torch.tensor(score.astype('float32')).to(device)

        return choice, image, (box, landmark, score)
"""


class MTCNNTrainDataset(torch.utils.data.Dataset):
    def __init__(self, root, data, transform=None, train=True, model='onet', batch_size=32):
        self.root = root
        self.transform = transform
        self.model = model

        self.box_images = data["box_images"]
        self.box_boxes = data["box_boxes"]
        self.negative_images = data["negative_images"]
        self.landmark_images = data["landmark_images"]
        self.landmark_landmarks = data["landmark_landmarks"]

        cutoff_ratio = 0.9

        #train and test split
        box_cutoff = int(len(self.box_images)*cutoff_ratio)
        landmark_cutoff = int(len(self.landmark_images)*cutoff_ratio)
        negative_cutoff = int(len(self.negative_images)*cutoff_ratio)

        if train:
            self.box_images = self.box_images[:box_cutoff]
            self.box_boxes = self.box_boxes[:box_cutoff]
            self.landmark_images = self.landmark_images[:landmark_cutoff]
            self.landmark_landmarks = self.landmark_landmarks[:landmark_cutoff]
            self.negative_images = self.negative_images[:negative_cutoff]
        else:
            self.box_images = self.box_images[box_cutoff:]
            self.box_boxes = self.box_boxes[box_cutoff:]
            self.landmark_images = self.landmark_images[landmark_cutoff:]
            self.landmark_landmarks = self.landmark_landmarks[landmark_cutoff:]
            self.negative_images = self.negative_images[negative_cutoff:]

        # normalizing box and landmark coordinates between 0 and 1
        self.box_boxes = self.box_boxes/48
        self.landmark_landmarks = self.landmark_landmarks/48

        # subtracting one from last two values in box
        self.box_boxes[:,2] -= 1
        self.box_boxes[:,3] -= 1

        self.all_images = self.preprocess_images(np.concatenate((self.box_images, self.landmark_images, self.negative_images)), model).astype(np.float32)
        self.all_labels = np.concatenate((np.array([[0.05, 0.95]]*len(self.box_images)), np.array([[0.05, 0.95]]*len(self.landmark_images)), np.array([[0.95, 0.05]]*len(self.negative_images)))).astype(np.float32)
        if (self.model == "pnet"):
          self.all_labels = np.reshape(self.all_labels, (-1, 2, 1, 1))
        self.len = len(self.all_images)

    @staticmethod
    def preprocess_images(images, model):
      #resize according to model
      if model == 'pnet':
        dim = 12
      elif model == 'rnet':
        dim = 24
      elif model == 'onet':
        dim = 48
      else:
        raise Exception("Invalid model name %s" % model)
      
      images_resized = np.zeros((len(images), dim, dim, 3))
      for i in range(len(images)):
        images_resized[i] = cv2.resize(images[i], (dim, dim))

      #transposing nhwc to nchw
      images_resized = np.transpose(images_resized, (0, 3, 1, 2))

      #normalize between 0 and 1
      return images_resized/255
    
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
       return self.all_images[idx], self.all_labels[idx]
