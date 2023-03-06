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
BATCH_SIZE_TRAIN = 10
BATCH_SIZE_TEST = 100
LR = 0.001


# torch.nn.functional.one_hot(torch.arange(0,5),10) 
#--- fixed constants ---
NUM_CLASSES = 1
NUM_IMAGES = 20000
img_dir = '../data/images'

#annotations_file = '../data/annotations/baby.txt'



#--- get images and make multi-hot-matrix ---
#label_names = ['baby','bird','car','clouds','dog','female','flower','male','night','people','portrait','river','sea','tree']
# NOTE NOTE NOTE TAKES THE FIRST LABEL FROM HERE, SO CHANGE THAT TO USE SOME OTHER
label_names = ['people','bird','car','clouds','dog','female','flower','male','night','people','portrait','river','sea','tree']
label_names = label_names[0:NUM_CLASSES]
one_hot = np.zeros([20000,NUM_CLASSES],dtype=int)
for c in range(0, NUM_CLASSES):
    annotations_file = '../data/annotations/' + label_names[c] + '.txt'            
    one_hot[np.loadtxt(annotations_file,dtype=int)-1,c] = 1 #  IS -1 NECESSARY??
#labels = np.loadtxt(annotations_file,dtype=int)-1

# create weights, used in lossfunction (maybe not useful?)
label_counts = sum(one_hot)
#WEIGHTS = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
WEIGHTS = torch.from_numpy(label_counts/NUM_IMAGES)



#https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, label_names, img_dir, transform=transforms.Grayscale(), target_transform=None):
        one_hot = np.zeros([NUM_IMAGES,NUM_CLASSES],dtype=int)
        for c in range(0, NUM_CLASSES):
            annotations_file = './data/annotations/' + label_names[c] + '.txt' 
            these_labels = np.loadtxt(annotations_file,dtype=int)-1
            sel_labels = these_labels[these_labels<NUM_IMAGES]
            one_hot[sel_labels,c] = 1 #  IS -1 NECESSARY??
        
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
                image = self.transform(image)       # image = self.transform(image) 
           
        im_arr = np.array(image)
        im_arr = im_arr.astype(np.float32)
        im_tensor = torch.from_numpy(im_arr)
        ##im_tensor = im_tensor.unsqueeze(0)
        #normalizer = transforms.Normalize(im_tensor.mean(),im_tensor.std())
        #im_tensor = normalizer(im_tensor)
        
        #resizer = torchvision.transforms.Resize((64,64))
        #im_tensor = resizer(im_tensor) 
        if self.target_transform:
            label = self.target_transform(label)
        return im_tensor, label


mydata = CustomImageDataset(label_names,img_dir)
# --- Dataset initialization ---
[train_set, dev_set] = torch.utils.data.random_split(mydata, [int(NUM_IMAGES*0.5), int(NUM_IMAGES*0.5)],generator=torch.Generator().manual_seed(42))

class_weights = [train_set.dataset.img_labels.sum(),len(train_set.dataset.img_labels)-train_set.dataset.img_labels.sum()]    

## reduce imbalance in training (only), show more images with rare labels
train_onehot = one_hot[train_set.indices]
label_counts_train = sum(train_onehot)
STIM_WEIGHT = np.array(train_onehot,dtype=int)
STIM_WEIGHT[STIM_WEIGHT==1] = class_weights[1]
STIM_WEIGHT[STIM_WEIGHT==0] = class_weights[0]

#STIM_WEIGHT = train_onehot*(1./label_counts_train)*1000
STIM_WEIGHT = np.sum(STIM_WEIGHT,axis=1)
STIM_WEIGHT = torch.from_numpy(STIM_WEIGHT)
sampler = torch.utils.data.WeightedRandomSampler(STIM_WEIGHT, num_samples=len(STIM_WEIGHT),replacement=True)


#train_loader = torch.utils.data.DataLoader(dataset=mydata, batch_size=BATCH_SIZE_TRAIN, shuffle=False)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=False,sampler=sampler)
train_loader2 = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=False)
#train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
#test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE_TEST, shuffle=False)
dev_loader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=BATCH_SIZE_TEST, shuffle=True)



class CNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
          nn.Conv2d(1,32,5, padding=1),              #nn.Conv2d(1,10,5),  32     
          #torch.nn.InstanceNorm2d(10),
          #nn.ReLU(), 
          #torch.nn.Dropout(p=0.3, inplace=False),
          nn.MaxPool2d(2,2),                 # nn.MaxPool2d(2,None),   
  
          # two layers
          nn.Conv2d(32,64,5,padding=1),              #nn.Conv2d(10,5,5,padding=0), 64
          #torch.nn.InstanceNorm2d(5),
          #nn.ReLU(), 
          #torch.nn.Dropout(p=0.3, inplace=False),
          nn.MaxPool2d(2,2),                 # nn.MaxPool2d(2,None),   
          
          nn.Conv2d(64,128,9,padding=1),               # nn.Conv2d(5,5,9,padding=1), 128
          #torch.nn.InstanceNorm2d(2),
          nn.ReLU(), 
          #torch.nn.Dropout(p=0.3, inplace=False),
          nn.MaxPool2d(2,2),                 # nn.MaxPool2d(2,None),   
          
          
        )
        self.layer2 = nn.Sequential(
            #nn.Linear(1960,1), 
            nn.Linear(128 * 12 * 12, 256),           # 256
            #nn.Linear(256,5),                  # nn.Linear(605,50),
            nn.Sigmoid(),
            nn.Linear(256,1),                    # nn.Linear(50,1),
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

model_list = []
optimizer_list = []
lossfunction_list = []
for m in range(NUM_CLASSES):
    this_model =  CNN().to(device)
    model_list.append(this_model)
    #optimizer = optim.SGD(model_list[m].parameters(), lr=LR,momentum=0.9)
    optimizer = optim.SGD(model_list[m].parameters(), lr=LR)
    optimizer_list.append(optimizer)
    loss_function = nn.BCELoss() #nn.BCEWithLogitsLoss(weight=WEIGHTS) #nn.MultiLabelSoftMarginLoss(reduction='none') #nn.BCEWithLogitsLoss()
    #loss-function = torchvision.ops.sigmoid_focal_loss()  BCELoss for binary classification
    lossfunction_list.append(loss_function)
    

  
    

#optimizer = optim.SGD(model.parameters(), lr=LR)

#optimizer = optim.Adam(model.parameters(), lr=LR)

dev_accuracies = np.zeros([1,N_EPOCHS])

#--- training ---
for epoch in range(N_EPOCHS):
   
    train_loss = torch.zeros(NUM_CLASSES)
    train_correct = torch.zeros(NUM_CLASSES)
    total = 0
    total_oneLabel = 0
    chance_correct = 0
    falsePositives = torch.zeros(NUM_CLASSES)
    falseNegatives = torch.zeros(NUM_CLASSES)
    if epoch > 5:
        train_loader = train_loader2
    for batch_num, (data, target) in enumerate(train_loader):
        #if batch_num > 15:
        #    break
        data, target = data.to(device), target.to(device)
        #target = target.byte()
        data = data.float() # THIS SHOULD BE MOVED TO INIT SOMEHOW, SOLVES THE 
        target = target.long()
        for m in range(NUM_CLASSES):
            model_list[m].zero_grad()
        
            prediction = model_list[m](data)
            this_target = torch.zeros(BATCH_SIZE_TRAIN,1)
            #this_target[:,0] = (target[:,m]-1)*-1 # not this label 
            #this_target[:,1] = target[:,m]
            this_target[:,0] = target[:,m]
            loss = lossfunction_list[m](prediction, this_target)
            #loss = torchvision.ops.sigmoid_focal_loss(prediction,this_target,alpha=1-WEIGHTS[m])
            #loss = torchvision.ops.sigmoid_focal_loss(prediction,this_target,alpha=(1-this_target.sum()/BATCH_SIZE_TRAIN))
            #loss = torchvision.ops.sigmoid_focal_loss(prediction,this_target,alpha=0.2)
            ##loss = loss.mean()
            # l2 regularization
            l2_lambda = 0.001
            l2_norm = sum(p.pow(2.0).sum()
                     for p in model_list[m].parameters())
            loss = loss + l2_lambda * l2_norm
            loss.backward()
            optimizer_list[m].step()
            #print(prediction.sum())
            #for name, param in model_list[m].named_parameters():
            #    print(name, param.grad.abs().sum())
            
            train_loss[m] = train_loss[m] + loss
            predLabel = prediction.round()
            #_, predLabel = torch.max(prediction, 1)
            #train_correct[m] += (predLabel == this_target[:,1]).sum().item()
            train_correct[m] += (predLabel == this_target).sum().item()
            #print(train_correct[m])
        
        
        #_, predLabel = torch.max(prediction, 1)
        total += target.size(0)*target.size(1)
        total_oneLabel += target.size(0)
 
        #prediction = (prediction * 0.7).round() # weight non-labels, but only affect accuracy, not model itself...
        
        
        
        # false positive and negative rates to see whether predicts only zeros
        falsePosNeg = predLabel - target
        falsePos = falsePosNeg > 0 
        falseNeg = falsePosNeg < 0 
        falsePositives += torch.count_nonzero(falsePos,dim=0)
        falseNegatives += torch.count_nonzero(falseNeg,dim=0)
        chance_correct += torch.count_nonzero(target-1).item()
        #print('target amounts %d' % target.sum())
        
        correct_over_chance = train_correct.sum() - chance_correct
        
            
        if batch_num < 10000: #== len(train_loader)-1:
            accuracies = 100.* train_correct / total_oneLabel
            #print(accuracies.round().tolist())
            #print(accuracies.mean())
            #print((100. * correct_over_chance/total).round().item())
            
            #print(*accuracies, sep = "%%, ") 
            #---print('Training: Epoch %d - Batch %d/%d: Loss: %.4f | Train Acc: %.3f%% (%d/%d)' % 
            #  (epoch, batch_num, len(train_loader), train_loss / (batch_num + 1), 
            #   100. * train_correct / total, train_correct, total))
            #print('Training: Epoch %d - Batch %d/%d: Loss: %.4f | Train Acc: %.3f%% (%d/%d)' % 
            #  (epoch, batch_num+1, len(train_loader), train_loss / (batch_num + 1), 
            #   100. * correct_over_chance / total, train_correct, total))
            #print('False Positives, false negatives %')
            false_pos = (100. * falsePositives / (total / NUM_CLASSES)).item()
            false_neg = (100. * falseNegatives / (total / NUM_CLASSES)).item()
            #print('False positives: %.2f%%, False negatives: %.2f%%' % (false_pos,false_neg))
            #print(100. * falseNegatives / (total / NUM_CLASSES))
            #---print('FPs: %.3f%%  FN:s: %.3f%% ' % ((100. * falsePositives.sum() / total).item(),
            #      (100. * falseNegatives.sum() / total).item()))
    print("+"*70)
    print('Training: Epoch %d - Batch %d/%d: Loss: %.4f | Train Acc: %.3f%% (%d/%d)' % 
              (epoch, batch_num, len(train_loader), train_loss / (batch_num + 1), 
               100. * train_correct / total, train_correct, total))        
    
    print('False positives: %.2f%%, False negatives: %.2f%%' % (false_pos,false_neg))


    # DEV TEST
    all_dev_acc = []
    dev_correct = torch.zeros(NUM_CLASSES)
    dev_total = 0
    chance_correct_dev = 0
    total_oneLabel = 0
    falsePositives = torch.zeros(NUM_CLASSES)
    falseNegatives = torch.zeros(NUM_CLASSES)
    for dev_batch_num, (dev_data, dev_target) in enumerate(dev_loader):
        if dev_batch_num > 10:
            break
        dev_data, dev_target = dev_data.to(device), dev_target.to(device)
        dev_data = dev_data.float()
        dev_target = dev_target.long()
        #print("*"*70)
        #print(dev_target)
        #print("+"*70)
        #print(dev_data)
        #print("*"*70)
        for m in range(NUM_CLASSES):
            
           
            total += dev_target.size(0)*dev_target.size(1)
            
            devPrediction = model_list[m].forward(dev_data)
            this_target = torch.zeros(BATCH_SIZE_TEST,1)

            this_target[:,0] = dev_target[:,m]
            predLabel = devPrediction.round()
            dev_correct[m] += (predLabel == this_target).sum().item()
        
        #_, devLabel = torch.max(devPrediction,1)

        
        total_oneLabel += dev_target.size(0)
        dev_total += dev_target.size(0)*dev_target.size(1)
        
        
        falsePosNeg = devPrediction - dev_target
        falsePos = falsePosNeg > 0 
        falseNeg = falsePosNeg < 0 
        falsePositives += torch.count_nonzero(falsePos,dim=0)
        falseNegatives += torch.count_nonzero(falseNeg,dim=0)
        
        chance_correct_dev += torch.count_nonzero(dev_target-1).item()
        #print('target amounts dev %d' % dev_target.sum())
        
        correct_over_chance = dev_correct.sum() - chance_correct_dev
        
        #if epoch =
        if dev_batch_num < 10000: #== len(dev_loader)-1:
            #print('dev')
            #accuracies = 100.* dev_correct / total_oneLabel
            #print(accuracies.round().tolist())
            #print(accuracies.mean())
            #print((100. * correct_over_chance/dev_total).round().item())
            #---print('Dev test: Epoch %d - Batch %d/%d: Dev Acc: %.3f%% (%d/%d)' % 
            #  (epoch, dev_batch_num+1, len(dev_loader), 
            #   100. * correct_over_chance / dev_total, dev_correct, dev_total))
            
            false_pos = (100. * falsePositives / (total / NUM_CLASSES)).item()
            false_neg = (100. * falseNegatives / (total / NUM_CLASSES)).item()
            #print('False Positives, false negatives %')
            #print(100. * falsePositives / (total / NUM_CLASSES))
            #print(100. * falseNegatives / (total / NUM_CLASSES))
            #print('FPs: %.3f%%  FN:s: %.3f%% ' % ((100. * falsePositives.sum() / dev_total).item(),
            #      (100. * falseNegatives.sum() / dev_total).item()))
    print("*"*70)
    print('Dev test: Epoch %d - Batch %d/%d: Dev Acc: %.3f%% (%d/%d) | Acc diff: %.3f%%' % 
      (epoch, dev_batch_num+1, len(dev_loader), 100 * dev_correct.sum() / dev_total, dev_correct, dev_total, 
       100. * correct_over_chance / dev_total)) 
    print('False positives: %.2f%%, False negatives: %.2f%%' % (false_pos,false_neg))       
    #dev_accuracies[0,epoch] = dev_correct / dev_total



