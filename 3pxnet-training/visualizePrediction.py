import torch
import numpy as np
import cv2

def drawBoxAndLandmarks(image, landmarks, box, landmarksColor=(255, 0, 0), boxColor=(255, 0, 0)):
  for i in range(5):
    landmark = (int(landmarks[i*2]*48), int(landmarks[i*2+1]*48))
    image = cv2.circle(image, landmark, 1, landmarksColor, 1)

  box_cv = [box[0], box[1], box[0]+box[2], box[1]+box[3]]
  box_cv = [int(elem*48) for elem in box_cv]
  box_cv = [(box_cv[0], box_cv[1]), (box_cv[2], box_cv[3])]
  image = cv2.rectangle(image, box_cv[0], box_cv[1], boxColor)

def showImage(index, box=None, landmarks=None, score=None, showLabels=True):
  image = all_image_data[index]
  image = np.copy(image)
  if showLabels:
    landmarksColor = (0, 0, 255)
    boxColor = (0, 0, 255)

    if score is None:
      score = all_score_data[index]
    if box is None:
      box = all_box_data[index]
      boxColor = (255, 0, 0)
    if landmarks is None:
      landmarks = all_landmark_data[index]
      landmarksColor = (255, 0, 0)

    drawBoxAndLandmarks(image, landmarks, box, landmarksColor, boxColor)
  
  image = cv2.resize(image, (192, 192))
  cv2_imshow(image)

  if showLabels:
    return score, box, landmarks

if model == 'pnet':
   trainset, testset, classes = utils_own.load_dataset('CELEBA_pnet')
   net = network.pnet(full=full, binary=binary)
elif model == 'rnet':
   trainset, testset, classes = utils_own.load_dataset('CELEBA_rnet')
   net = network.rnet(full=full, binary=binary)
elif model == 'onet':
   trainset, testset, classes = utils_own.load_dataset('CELEBA_onet')
   net = network.onet(full=full, binary=binary)
else:
   raise Exception("Invalid model name %s" % model)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

index = 0
sample = trainloader[index]

def visualize():
	#NCHW
	#index = int(np.random.random()*len(all_box_data))
	index = 182534

	print("Original image")
	showImage(index, showLabels=False)

	print("Ground truth labels")
	score, box, landmarks = showImage(index)

	p_net_binarized = torch.load("p_net_model_binary.pt")

	#input_tensor = np.zeros((1, 3, 12, 12)).astype('float32')
	#output = p_net_binarized(torch.tensor(input_tensor))
	models = [p_net_binarized]
	for i in range(len(models)):
	  model = models[i]

	  predictedBox = None
	  predictedLandmarks = None
	  predictedScore = None

	  if i == 0:
	    print("p-net prediction")
	    image_12_12 = np.expand_dims(all_image_data_12_12[index], 0)
	    image_12_12_torch = torch.tensor(np.transpose(image_12_12, (0, 3, 1, 2)))
	    prediction = model(image_12_12_torch)
	    predictedBox = prediction[0][0][0][0]
	    predictedScore = prediction[1][0][0][0]
	  elif i == 1:
	    print("r-net prediction")
	    image_24_24 = all_image_data_24_24[index]
	    image_24_24_torch = torch.tensor(np.transpose(image_24_24, (0, 3, 1, 2)))
	    prediction = model(np.expand_dims(image_24_24_torch, 0))
	    predictedBox = prediction[0][0]
	    predictedScore = prediction[1][0]
	  elif i == 2:
	    print("o-net prediction")
	    image = all_image_data[index]
	    image_torch = torch.tensor(np.transpose(image, (0, 3, 1, 2)))
	    prediction = model(np.expand_dims(image_torch, 0))
	    predictedLandmarks = prediction[2][0]
	    predictedBox = prediction[0][0]
	    predictedLandmarks = prediction[1][0]
	    predictedScore = prediction[2][0]
	  
	  #print("Predicted score of", predictedScore[0])
	  #print("Predicted box", predictedBox)
	  #print("Predicted landmarks", predictedLandmarks)
	  
	  _, _, _ = showImage(index, box=predictedBox, landmarks=predictedLandmarks, score=predictedScore)

	#print("Score is", score, "Predicted score is", predictedScore)
	#print("Box is", box, "Predicted box is", predictedBox)
	#print("Landmarks are", landmarks, "Predicted landmarks are", predictedLandmarks)

if __name__ == '__main__':
	visualize()