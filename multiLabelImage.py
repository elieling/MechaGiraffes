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

#--- hyperparameters ---
N_EPOCHS = 50   
BATCH_SIZE_TRAIN = 100
BATCH_SIZE_TEST = 100
LR = 0.01


# torch.nn.functional.one_hot(torch.arange(0,5),10) 
#--- fixed constants ---
NUM_CLASSES = 24
NUM_IMAGES = 20000
img_dir = '../data/images'
annotations_file = '../data/annotations/baby.txt'
import os
import pandas as pd
from torchvision.io import read_image

#https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, img_dir, transform=transforms.Grayscale(), target_transform=None):
        one_hot = np.zeros(20000,dtype=int)
        one_hot[np.loadtxt(annotations_file,dtype=int)-1] = 1 #  IS -1 NECESSARY??
        self.img_labels = one_hot
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        #img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]) # when transfomrmed to dataframe with multiple columns..?
        im_name = 'im%d.jpg' % idx
        img_path = os.path.join(self.img_dir, im_name)
        image = read_image(img_path)
        
        #label = self.img_labels.iloc[idx, 1]
        label = self.img_labels[idx]
        if self.transform:
            if torch.Tensor.size(image,0) == 3:
                image = self.transform(image)
           
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


mydata = CustomImageDataset(annotations_file,img_dir)
# --- Dataset initialization ---
[train_set, dev_set] = torch.utils.data.random_split(mydata, [int(NUM_IMAGES*0.5), int(NUM_IMAGES*0.5)],generator=torch.Generator().manual_seed(42))


train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=False)
#test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE_TEST, shuffle=False)
dev_loader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=BATCH_SIZE_TEST, shuffle=True)



class CNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CNN, self).__init__()
       # self.layer1 = nn.Sequential(
       #   nn.Conv2d(1,32,3),
       ##   torch.nn.InstanceNorm2d(32),
        #  nn.ReLU(), 
        #  torch.nn.Dropout(p=0.35, inplace=False),
        #  nn.MaxPool2d(2,None),
          
          #nn.Conv2d(32,24,5),
          #torch.nn.InstanceNorm2d(32),
          #nn.ReLU(), 
          #torch.nn.Dropout(p=0.35, inplace=False),
          #nn.MaxPool2d(2,None)
       # )
        self.layer2 = nn.Sequential(
            nn.Linear(128,NUM_CLASSES)
            #torch.nn.Dropout(p=0.35, inplace=False)
        )
        
    def forward(self, x):
        #x = self.layer1(x)
       # x = torch.flatten(x, 1)
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
loss_function = nn.CrossEntropyLoss()
dev_accuracies = np.zeros([1,N_EPOCHS])

#--- training ---
for epoch in range(N_EPOCHS):
    train_loss = 0
    train_correct = 0
    total = 0
    for batch_num, (data, target) in enumerate(train_loader):
        model.zero_grad()
        data, target = data.to(device), target.to(device)
        #target = target.byte()
        data = data.float() # THIS SHOULD BE MOVED TO INIT SOMEHOW, SOLVES THE 
        # PROBLEM OF EXPECTED TYPE BYTE/FLOAT PROBLEM
        prediction = model(data)
        loss = loss_function(prediction, target)
        
        # l2 regularization
        l2_lambda = 0.001
        l2_norm = sum(p.pow(2.0).sum()
                 for p in model.parameters())
        loss = loss + l2_lambda * l2_norm
        
        train_loss = train_loss + loss
        loss.backward()
        optimizer.step()
        
        _, predLabel = torch.max(prediction, 1)
        total += target.size(0)
        train_correct += (predLabel == target).sum().item()
            
       # if batch_num == len(train_loader)-1:
       #     print('Training: Epoch %d - Batch %d/%d: Loss: %.4f | Train Acc: %.3f%% (%d/%d)' % 
       #       (epoch, batch_num, len(train_loader), train_loss / (batch_num + 1), 
       #        100. * train_correct / total, train_correct, total))
    
    
    # WRITE CODE HERE
    all_dev_acc = []
    dev_correct = 0
    dev_total = 0
    for dev_batch_num, (dev_data, dev_target) in enumerate(dev_loader):
        dev_data, target = dev_data.to(device), dev_target.to(device)
        devPrediction = model.forward(dev_data)
        _, devLabel = torch.max(devPrediction,1)
        
        
        
        dev_total += target.size(0)
        dev_correct += (devLabel == target).sum().item()
        
        #if epoch =
        if dev_batch_num == len(dev_loader)-1:
            print('Dev test: Epoch %d - Batch %d/%d: Dev Acc: %.3f%% (%d/%d)' % 
              (epoch, dev_batch_num, len(dev_loader), 
               100. * dev_correct / dev_total, dev_correct, dev_total))
    dev_accuracies[0,epoch] = dev_correct / dev_total



