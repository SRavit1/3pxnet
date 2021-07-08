import os
import torch
import numpy as np
import cv2

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
            raise Exception("inappropriate model name")

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