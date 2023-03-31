import torch
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms, datasets, models
from torchvision.io import read_image
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

from PIL import Image

#--- hyperparameters ---
N_EPOCHS = 6   
BATCH_SIZE_TRAIN = 50
BATCH_SIZE_TEST = 100
LR = 0.001



label_names2 = ['baby','bird','car','clouds','dog','female','flower','male','night','people','portrait','river','sea','tree']
classN = 14
NUM_CLASSES = 1
NUM_IMAGES = 20000
img_dir = '../data/images'

# the (slightly modified) priors for the pseudo-Bayesian test comparison in an 
# unbalanced set
stim_priors = np.array([0.99525, 0.972, 0.987, 0.92, 0.9776, 0.831, 0.942,
                        0.85105, 0.98, 0.61985, 0.83395, 0.996, 0.995, 0.97375 ])

for mm in range(classN):

    label_names = ['baby','bird','car','clouds','dog','female','flower','male','night','people','portrait','river','sea','tree']
    label_names = label_names[mm:mm+1]
    one_hot = np.zeros([20000,NUM_CLASSES],dtype=int)
    for c in range(0, NUM_CLASSES):
        annotations_file = '../data/annotations/' + label_names[c] + '.txt'            
        one_hot[np.loadtxt(annotations_file,dtype=int)-1,c] = 1 #  IS -1 NECESSARY??
    # create weights, used in lossfunction (maybe not useful?)
    label_counts = sum(one_hot)
    #WEIGHTS = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    #WEIGHTS = torch.from_numpy(label_counts/NUM_IMAGES)
    
    
    
    # from https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
    class CustomImageDataset(torch.utils.data.Dataset):
        def __init__(self, label_names, img_dir, transform=transforms.Grayscale(), target_transform=None):
            one_hot = np.zeros([NUM_IMAGES,NUM_CLASSES],dtype=int)
            for c in range(0, NUM_CLASSES):
                annotations_file = '../data/annotations/' + label_names[c] + '.txt' 
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
            #image = read_image(img_path)
            image = Image.open(img_path).convert('RGB')
            #image = torchvision.transforms.functional.adjust_sharpness(image, 3)
            
            
            transform2 = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.4),
                torchvision.transforms.RandAugment(),
                torchvision.transforms.RandomAdjustSharpness(2, p=0.2),
                torchvision.transforms.RandomRotation(5)
            ])
            image = transform2(image)
            label = self.img_labels[idx]
            #if self.transform:
            #    if torch.Tensor.size(image,0) == 3:
            #        image = self.transform(image)
               
            totensor = torchvision.transforms.ToTensor()
            im_tensor = totensor(image)
            resizer = torchvision.transforms.Resize((224,224))
            im_tensor = resizer(im_tensor) 
            if self.target_transform:
                label = self.target_transform(label)
            return im_tensor, label
    
    
    mydata = CustomImageDataset(label_names,img_dir)
    # --- Dataset initialization ---
    [train_set, dev_set] = torch.utils.data.random_split(mydata, [int(NUM_IMAGES*0.8), int(NUM_IMAGES*0.2)],generator=torch.Generator().manual_seed(42))
    
    ## reduce imbalance in training (only), show more images with rare labels
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
    dev_loader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=BATCH_SIZE_TEST, shuffle=True)
    
    # Resnet model
    model_ft = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model_ft.parameters():
        param.requires_grad = False
    num_ftrs = model_ft.fc.in_features
    
    model_ft.fc = nn.Linear(num_ftrs, 1)
    input_size = 224
    
    #--- set up ---
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    TheModel =  model_ft.to(device)

    optimizer = optim.SGD(TheModel.parameters(), lr=LR)
    lossfunction = nn.BCEWithLogitsLoss()    
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
        
        for batch_num, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data = data.float()
            target = target.long()
            TheModel.zero_grad()
            prediction = TheModel(data)
            this_target = torch.zeros(BATCH_SIZE_TRAIN,1)
            this_target[:,0] = target[:,0]
            loss = lossfunction(prediction, this_target)
            
            # l2 regularization
            l2_lambda = 0.001
            l2_norm = sum(p.pow(2.0).sum()
                     for p in TheModel.parameters())
            
            loss = loss + l2_lambda * l2_norm
            loss.backward()
            optimizer.step()
            prediction = torch.sigmoid(prediction) # for interpretation
            print(prediction.sum())
            #for name, param in model_list[m].named_parameters():
            #    print(name, param.grad.abs().sum())
            
            train_loss = train_loss + loss
            predLabel = prediction.round()
            
            
            train_correct += (predLabel == this_target).sum().item()
            total += target.size(0)*target.size(1)
            total_oneLabel += target.size(0)
            
            # false positive and negative rates to see whether predicts only zeros
            falsePosNeg = predLabel - target
            falsePos = falsePosNeg > 0 
            falseNeg = falsePosNeg < 0 
            falsePositives += torch.count_nonzero(falsePos,dim=0)
            falseNegatives += torch.count_nonzero(falseNeg,dim=0)
            chance_correct += torch.count_nonzero(target-1).item()
            print('target amounts %d' % target.sum())
            
            correct_over_chance = train_correct.sum() - chance_correct
            
                
            if batch_num < 10000: #== len(train_loader)-1:
                accuracies = 100.* train_correct / total_oneLabel
                
                print('Training: Epoch %d - Batch %d/%d: Loss: %.4f | Train Acc: %.3f%% (%d/%d)' % 
                  (epoch, batch_num, len(train_loader), train_loss / (batch_num + 1), 
                   100. * train_correct / total, train_correct, total))
                
                false_pos = (100. * falsePositives / (total / NUM_CLASSES)).item()
                false_neg = (100. * falseNegatives / (total / NUM_CLASSES)).item()
                print('False positives: %.2f%%, False negatives: %.2f%%' % (false_pos,false_neg))
                
        
        
        # DEV TEST
        all_dev_acc = []
        dev_correct = torch.zeros(NUM_CLASSES)
        dev_total = 0
        chance_correct_dev = 0
        total_oneLabel = 0
        truePositives = torch.zeros(NUM_CLASSES)
        targetPositives = torch.zeros(NUM_CLASSES)
        falsePositives = torch.zeros(NUM_CLASSES)
        falseNegatives = torch.zeros(NUM_CLASSES)
        for dev_batch_num, (dev_data, dev_target) in enumerate(dev_loader):
            if dev_batch_num > 2:
                break
            dev_data, dev_target = dev_data.to(device), dev_target.to(device)
            dev_data = dev_data.float()
            dev_target = dev_target.long()
        
            devPrediction = TheModel.forward(dev_data)
            this_target = torch.zeros(BATCH_SIZE_TEST,1)
            this_target[:,0] = dev_target[:,0]
            devPrediction = torch.sigmoid(devPrediction)
            #predLabel = devPrediction.round()
            predLabel = (devPrediction>stim_priors[mm]).int()
            dev_correct += (predLabel == dev_target).sum().item()

            total_oneLabel += dev_target.size(0)
            dev_total += dev_target.size(0)*dev_target.size(1)
            
            
            falsePosNeg = predLabel - dev_target
            falsePos = falsePosNeg > 0 
            falseNeg = falsePosNeg < 0 
            falsePositives += torch.count_nonzero(falsePos,dim=0)
            falseNegatives += torch.count_nonzero(falseNeg,dim=0)
            truePositives += torch.count_nonzero(dev_target[predLabel==1] == 1)
            targetPositives += (dev_target==1).sum()
            chance_correct_dev += torch.count_nonzero(dev_target-1).item()
            correct_over_chance = dev_correct.sum() - chance_correct_dev
            
            if dev_batch_num < 10000: 
                false_pos = (100. * falsePositives / (dev_total / NUM_CLASSES)).item()
                false_neg = (100. * falseNegatives / (dev_total / NUM_CLASSES)).item()
                true_pos = truePositives
                print('Dev test: Epoch %d - Batch %d/%d: Dev Acc: %.3f%% (%d/%d)' % 
                  (epoch, dev_batch_num+1, len(dev_loader), 
                   100. * correct_over_chance / dev_total, dev_correct, dev_total))
                print('False positives: %.2f%%, False negatives: %.2f%%, True positives: %.0f/%.0f' % (false_pos,false_neg,true_pos, targetPositives))       
        
        if epoch%5==0:
            for m in range(NUM_CLASSES):
                file_name = "model" + str(m+1) + label_names[m] + "epoch" + str(epoch) + ".pth"
                torch.save(TheModel.state_dict(), file_name)
    
    