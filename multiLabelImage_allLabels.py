import torch
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from torchvision.io import read_image
from sklearn.metrics import confusion_matrix, precision_score

#--- hyperparameters ---
N_EPOCHS = 50            # 50   
BATCH_SIZE_TRAIN = 100
BATCH_SIZE_TEST = 100
LR = 0.01


# torch.nn.functional.one_hot(torch.arange(0,5),10) 
#--- fixed constants ---
NUM_CLASSES = 14
NUM_IMAGES = 20000
img_dir = './data/images'               # .
#annotations_file = '../data/annotations/baby.txt'
label_names = ['baby','bird','car','clouds','dog','female','flower','male','night','people','portrait','river','sea','tree']
one_hot = np.zeros([20000,NUM_CLASSES],dtype=int)
for c in range(0, NUM_CLASSES):
    annotations_file = './data/annotations/' + label_names[c] + '.txt'            
    one_hot[np.loadtxt(annotations_file,dtype=int)-1,c] = 1 #  IS -1 NECESSARY??
#labels = np.loadtxt(annotations_file,dtype=int)-1

# create weights, used in lossfunction (maybe not useful?)
label_counts = sum(one_hot)
#WEIGHTS = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
WEIGHTS = torch.from_numpy(label_counts/NUM_IMAGES)

import os
import pandas as pd
from torchvision.io import read_image

#https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, label_names, img_dir, transform=transforms.Grayscale(), target_transform=None):
        
        one_hot = np.zeros([20000,NUM_CLASSES],dtype=int)
        for c in range(0, NUM_CLASSES-1):
            annotations_file = './data/annotations/' + label_names[c] + '.txt'                      # .   
            one_hot[np.loadtxt(annotations_file,dtype=int)-1,c] = 1 #  IS -1 NECESSARY??
        #labels = np.loadtxt(annotations_file,dtype=int)-1
        self.img_labels = one_hot
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        #img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]) # when transfomrmed to dataframe with multiple columns..?
        im_name = 'im%s.jpg' % str(idx+1)
        img_path = os.path.join(self.img_dir, im_name)
        image = read_image(img_path)
        
        #label = self.img_labels.iloc[idx, 1]
        label = self.img_labels[idx]

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])

        if self.transform:
            if torch.Tensor.size(image,0) == 3:
                image = self.transform(image)

        im_arr = np.array(image)
        im_arr = im_arr.astype(np.float32)
        im_tensor = torch.from_numpy(im_arr)
           
        if self.target_transform:
            label = self.target_transform(label)
        return im_tensor, label                     # image


mydata = CustomImageDataset(label_names,img_dir)
# --- Dataset initialization ---
[train_set, dev_set] = torch.utils.data.random_split(mydata, [int(NUM_IMAGES*0.5), int(NUM_IMAGES*0.5)],generator=torch.Generator().manual_seed(42))

class_weights = [train_set.dataset.img_labels.sum(),len(train_set.dataset.img_labels)-train_set.dataset.img_labels.sum()]    

train_onehot = one_hot[train_set.indices]
label_counts_train = sum(train_onehot)
STIM_WEIGHT = np.array(train_onehot,dtype=int)
STIM_WEIGHT[STIM_WEIGHT==1] = class_weights[1]
STIM_WEIGHT[STIM_WEIGHT==0] = class_weights[0]

STIM_WEIGHT = np.sum(STIM_WEIGHT,axis=1)
STIM_WEIGHT = torch.from_numpy(STIM_WEIGHT)
sampler = torch.utils.data.WeightedRandomSampler(STIM_WEIGHT, num_samples=len(STIM_WEIGHT),replacement=True)

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=False,sampler=sampler)
train_loader2 = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=False)
#test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE_TEST, shuffle=False)
dev_loader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=BATCH_SIZE_TEST, shuffle=True)



class CNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
          nn.Conv2d(1,16,5),                 #5
          #torch.nn.InstanceNorm2d(32),
          nn.ReLU(), 
          #torch.nn.Dropout(p=0.35, inplace=False),
          nn.MaxPool2d(2,None),
          nn.Conv2d(16,32,5,padding=1),  
          nn.MaxPool2d(2,None),  
          nn.Conv2d(32,64,9,padding=1),   
          nn.ReLU(), 
          nn.MaxPool2d(2,None),      
          
          #nn.Conv2d(32,24,5),
          #torch.nn.InstanceNorm2d(32),
          #nn.ReLU(), 
          #torch.nn.Dropout(p=0.35, inplace=False),
          #nn.MaxPool2d(2,None)
        )
        self.layer2 = nn.Sequential(
            #nn.Linear(19220,14)
            #torch.nn.Dropout(p=0.35, inplace=False)
            nn.Linear(64 * 12 * 12, 128),           # 256
            #nn.Linear(256,5),                  # nn.Linear(605,50),
            nn.Sigmoid(),
            nn.Linear(128,14),                    # nn.Linear(50,1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.layer1(x)
        x = torch.flatten(x, 1)
        x = self.layer2(x)
        
        return x
        
#--- set up ---
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = CNN().to(device)

# WRITE CODE HERE
#optimizer = optim.SGD(model.parameters(), lr=LR)
optimizer = optim.SGD(model.parameters(), lr=LR,momentum=0.9)
#optimizer = optim.Adam(model.parameters(), lr=LR)
loss_function = nn.BCEWithLogitsLoss()
dev_accuracies = np.zeros([1,N_EPOCHS])

#--- training ---
for epoch in range(N_EPOCHS):
    train_loss = 0
    train_correct = 0
    #train_loss = torch.zeros(NUM_CLASSES)
    #train_correct = torch.zeros(NUM_CLASSES)
    total = 0
    total_oneLabel = 0
    chance_correct = 0
    falsePositives = torch.zeros(NUM_CLASSES)
    falseNegatives = torch.zeros(NUM_CLASSES)
    if epoch == 5:
        train_loader = train_loader2
    for batch_num, (data, target) in enumerate(train_loader):
        model.zero_grad()
        data, target = data.to(device), target.to(device)
        #target = target.byte()
        data = data.float() 
        target = target.float() # THIS SHOULD BE MOVED TO INIT SOMEHOW, SOLVES THE 
        # PROBLEM OF EXPECTED TYPE BYTE/FLOAT PROBLEM
        prediction = model(data)
        #target = target.long()
        #target = target.float()
        loss = loss_function(prediction, target)
        # l2 regularization
        l2_lambda = 0.001
        l2_norm = sum(p.pow(2.0).sum()
                 for p in model.parameters())
        loss = loss + l2_lambda * l2_norm
        
        train_loss = train_loss + loss
        loss.backward()
        optimizer.step()
        
        #_, predLabel = torch.max(prediction, 1)
        total += target.size(0)*target.size(1)
        
        # this is silly way to decide which values are predicting that is
        #this label and which that isn't, NEED TO BE FIXED
        outmap_min, _ = torch.min(prediction, dim=1, keepdim=True)
        outmap_max, _ = torch.max(prediction, dim=1, keepdim=True)
        prediction = (prediction - outmap_min) / (outmap_max - outmap_min)
        prediction = prediction.round()
        train_correct += (prediction == target).sum().item()

        falsePosNeg = prediction - target
        falsePos = falsePosNeg > 0 
        falseNeg = falsePosNeg < 0 
        falsePositives += torch.count_nonzero(falsePos,dim=0)
        falseNegatives += torch.count_nonzero(falseNeg,dim=0)
        chance_correct += torch.count_nonzero(target-1).item()

        correct_over_chance = train_correct - chance_correct
            
        #if batch_num == len(train_loader)-1:
        #print('Training: Epoch %d - Batch %d/%d: Loss: %.4f | Train Acc: %.3f%% (%d/%d)' % 
        #      (epoch, batch_num, len(train_loader), train_loss / (batch_num + 1), 
        #       100. * train_correct / total, train_correct, total))
        #print((epoch, batch_num+1, len(train_loader), train_loss / (batch_num + 1), 
        #   100. * train_correct / total, train_correct, total, 100. * correct_over_chance / total))
        
    print('Training: Epoch %d - Batch %d/%d: Loss: %.4f | Train Acc: %.3f%% (%d/%d) | Correct over chance: %.3f%%' % 
          (epoch, batch_num+1, len(train_loader), train_loss / (batch_num + 1), 
           100. * train_correct / total, train_correct, total, 100. * correct_over_chance / total))
    print('FPs: %.3f%%  FN:s: %.3f%% ' % ((100. * falsePositives.sum() / total).item(),
              (100. * falseNegatives.sum() / total).item()))
    
    
    # WRITE CODE HERE
    all_dev_acc = []
    dev_correct = 0
    #dev_correct = torch.zeros(NUM_CLASSES)
    dev_total = 0
    chance_correct_dev = 0
    total_oneLabel = 0
    falsePositives = torch.zeros(NUM_CLASSES)
    falseNegatives = torch.zeros(NUM_CLASSES)
    total_precision = 0
    for dev_batch_num, (dev_data, dev_target) in enumerate(dev_loader):
        dev_data, dev_target = dev_data.to(device), dev_target.to(device)
        dev_data = dev_data.float()
        devPrediction = model.forward(dev_data)
        #target = dev_target.long()
        target = dev_target.float()
        #_, devLabel = torch.max(devPrediction,1)
        
        outmap_min, _ = torch.min(devPrediction, dim=1, keepdim=True)
        outmap_max, _ = torch.max(devPrediction, dim=1, keepdim=True)
        devPrediction = (devPrediction - outmap_min) / (outmap_max - outmap_min)
        devPrediction = devPrediction.round()
        
        
        dev_total += target.size(0)*target.size(1)
        dev_correct += (devPrediction == target).sum().item()

        falsePosNeg = devPrediction - dev_target
        falsePos = falsePosNeg > 0 
        falseNeg = falsePosNeg < 0 
        falsePositives += torch.count_nonzero(falsePos,dim=0).sum()
        falseNegatives += torch.count_nonzero(falseNeg,dim=0).sum()
        
        chance_correct_dev += torch.count_nonzero(dev_target-1).item()
        correct_over_chance = dev_correct - chance_correct_dev

        #false_pos = (100. * falsePositives / (total / NUM_CLASSES)).item()
        #false_neg = (100. * falseNegatives / (total / NUM_CLASSES)).item()

        #if epoch =
        #if dev_batch_num == len(dev_loader)-1:
        #print('Dev test: Epoch %d - Batch %d/%d: Dev Acc: %.3f%% (%d/%d)' % 
        #      (epoch, dev_batch_num, len(dev_loader), 
        #       100. * dev_correct / dev_total, dev_correct, dev_total))
        ##################################################################################################
        ############################################ TEST BOX ############################################
        ##################################################################################################
        #devPrediction = model(dev_data)
        #loss = loss_function(devPrediction, target)
        #loss.backward()
        #optimizer.step()
        #print(devPrediction.shape)
        #print(devPrediction)
        #y_pred_labels = torch.argmax(devPrediction, dim=1)  
        #y_pred_np = y_pred_labels.numpy()  
        #y_true_np = target.numpy() 
        #print(y_pred_np.shape, y_true_np.shape)
        #print(y_pred_np, y_true_np)
        #precision = precision_score(y_true_np, y_pred_np, average='weighted')
        #total_precision += precision
        #print("Precision:", precision,"---- Total precision:", total_precision)
        ##################################################################################################
        ##################################################################################################
        ##################################################################################################

    #dev_accuracies[0,epoch] = dev_correct / dev_total
    print("*"*70)
    #print(falsePositives[0], "---", total)
    print('Dev test: Epoch %d - Batch %d/%d: Dev Acc: %.3f%% (%d/%d) | Acc diff: %.3f%%' % 
      (epoch, dev_batch_num+1, len(dev_loader), 100 * dev_correct / dev_total, dev_correct, dev_total, 
       100. * correct_over_chance / dev_total)) 
    #print('False positives: %.2f%%, False negatives: %.2f%%' % (false_pos,false_neg))



