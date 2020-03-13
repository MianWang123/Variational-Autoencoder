"""
CIS522-Deep Learning for Data Science: Convolutional Autoencoder
Author: Mian Wang  
Time: 3/2/20
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision as tv
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from glob import glob
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler
from torchsummary import summary

# loading google drive from google colab
import os
from google.colab import drive
drive.mount('/content/drive')

# upload the datasets.7z to my drive, and unzip in colab
DATASETS_PATH = '/content/drive/My Drive/datasets.7z'
ESCAPED_PATH = DATASETS_PATH.replace(" ", "\\ ") 
!7z x {ESCAPED_PATH}


# Step 0: Set GPU in google colab and launch tensorboard
device = torch.device('cuda:0' if torch.cuda.is_available() else 'gpu')
%load_ext tensorboard


# Step 1: Import the dataset from 5 different folders
UT_transforms = transforms.Compose([ transforms.Resize((100, 100)),
                                     transforms.ToTensor()        ])
UT_dataset = dset.ImageFolder(root='DATASETS/UTZappos50K', transform=UT_transforms)


# Step 2: Divide data into training set, validation set, and test set(7:2:1), then set batch size to 64
idx = list(range(len(UT_dataset)))
np.random.shuffle(idx)
split1, split2 = int(np.floor(len(idx)*0.7)), int(np.floor(len(idx)*0.9))
ut_train_sampler = SubsetRandomSampler(idx[:split1])
ut_val_sampler = SubsetRandomSampler(idx[split1:split2])
ut_test_sampler = SubsetRandomSampler(idx[split2:])

batch_size = 64
ut_train_loader = DataLoader(UT_dataset, batch_size=batch_size, sampler=ut_train_sampler)
ut_val_loader = DataLoader(UT_dataset, batch_size=batch_size, sampler=ut_val_sampler)
ut_test_loader = DataLoader(UT_dataset, batch_size=batch_size, sampler=ut_test_sampler)


# Step 3: Establish CAE network
class CAE(nn.Module):
  def __init__(self):
    super(CAE, self).__init__()
    self.encoder = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=16, kernel_size=2, stride=2, padding=0),
        nn.ReLU(),             
        #nn.MaxPool2d(2, stride=2),            
        nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=3, padding=0),     
        nn.ReLU(),          
        #nn.MaxPool2d(2, stride=2),           
        nn.Tanh()    )
    self.decoder = nn.Sequential(
        nn.ConvTranspose2d(in_channels=8, out_channels=12, kernel_size=5, stride=2, padding=1),       
        nn.ReLU(),             
        nn.ConvTranspose2d(in_channels=12, out_channels=3, kernel_size=4, stride=3, padding=0),
        nn.Sigmoid()   )       

  def forward(self, x):
    z = self.encoder(x)
    recon_x = self.decoder(z)
    return recon_x   
  
model_cae = CAE().to(device)


# Step 4: Set optimizer and loss function, use summary to check the network
optimizer = torch.optim.Adam(model_cae.parameters(), lr=0.001)
criterion = nn.MSELoss()
summary(model_cae, (3,100,100))


# Step 5: Train data with cae model, send training loss to tensorboard for supervision
epochs = 20
logger = SummaryWriter('logs/CAE')
for epoch in range(epochs):
  Loss = 0
  for i, (x,_) in enumerate(ut_train_loader):
    x = x.to(device)
    optimizer.zero_grad()
    recon_x = model_cae(x)
    loss = criterion(recon_x, x)
    loss.backward()
    optimizer.step()
    Loss += loss.item()

  logger.add_scalar('training loss', Loss, epoch)
  print(f'Epoch: {epoch+1} Training Loss: {Loss}')

  
# Step 6: Randomly pick 5 images from the dataset, use the trained model to generate fake images
indices = np.random.choice(len(UT_dataset), 5, replace=False)
original_imgs = torch.empty([5,3,100,100], dtype=torch.float32)
reconstruct_imgs = torch.empty([5,3,100,100], dtype=torch.float32)

count = 0
for i in indices:
  original_imgs[count] = UT_dataset[i][0] 
  x = UT_dataset[i][0].unsqueeze(0).to(device)
  z = model_cae.encoder(x)
  recon_x = model_cae.decoder(z) 
  reconstruct_imgs[count] = recon_x.data.cpu()
  count += 1
  
# print original and generated images
plt.figure(figsize=(10,15))
for j in range(5):
  plt.subplot(5,2,2*j+1)
  plt.imshow(original_imgs[j].permute(1,2,0))
  plt.subplot(5,2,2*j+2)
  plt.imshow(reconstruct_imgs[j].permute(1,2,0))
plt.savefig('cae.png')
