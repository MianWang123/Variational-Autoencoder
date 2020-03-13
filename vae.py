"""
CIS522-Deep Learning for Data Science: Variational Autoencoder
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


# Step 3: Establish VAE network (encoder, reparametrizaion, decoder)
class VAE(nn.Module):
  def __init__(self):
    super(VAE, self).__init__()

    self.encoder = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=12, kernel_size=2, stride=2, padding=0),
        nn.ReLU(),
        nn.BatchNorm2d(12),
        nn.Conv2d(in_channels=12, out_channels=24, kernel_size=4, stride=2, padding=0),
        nn.ReLU(),
        nn.BatchNorm2d(24),
        nn.Conv2d(in_channels=24, out_channels=6, kernel_size=3, stride=3, padding=0),
        nn.Tanh()        )
    
    self.l1 = nn.Linear(6*8*8, 256)
    self.l2 = nn.Linear(6*8*8, 256)
    self.l3 = nn.Linear(256, 6*8*8)

    self.decoder = nn.Sequential(
        nn.ConvTranspose2d(in_channels=6, out_channels=12, kernel_size=2, stride=2, padding=0),
        nn.ReLU(),
        nn.BatchNorm2d(12),
        nn.ConvTranspose2d(in_channels=12, out_channels=24, kernel_size=3, stride=2, padding=0),
        nn.ReLU(),
        nn.BatchNorm2d(24),
        nn.ConvTranspose2d(in_channels=24, out_channels=3, kernel_size=4, stride=3, padding=0),
        nn.Sigmoid()      )
  
  def reparametrize(self, mu, log_var):
    std = torch.exp(log_var/2)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)

  def forward(self, x):
    x = self.encoder(x)   
    x = x.view(x.size(0), -1)
    mu, log_var = self.l1(x), self.l2(x)
    z = self.reparametrize(mu, log_var)
    z = self.l3(z)
    recon_x = self.decoder(z.view(-1,6,8,8))
    return recon_x, mu, log_var
    
model_vae = VAE().to(device)


# Step 4: Pick Adam as optimizer, use summary to check the network
optimizer = torch.optim.Adam(model_vae.parameters(), lr=0.001)
summary(model_vae, (3,100,100))


# Step 5: Define loss function (BCE+KLD), the ratio 0.1 was added during hyperparameter tunning.
def loss_func(recon_x, x, mu, log_var):
  # reconstruction loss
  RCL = F.binary_cross_entropy(recon_x, x, reduction='sum')
  
  # KL divergence loss
  KLD = 0.5 * torch.sum(mu**2 + log_var.exp() - 1 - log_var)
  return RCL+0.1*KLD, RCL, KLD


# Step 6: Displays the images in a (rows * cols) grid
def generate_images(imgs, rows, cols, name=None, show=True):
  imgs = imgs[:rows*cols].cpu().detach()
  fig, ax = plt.subplots(rows, cols)
  count = 0
  imgs = np.reshape(imgs, (imgs.shape[0],3,100,100))
  imgs = np.clip(imgs, a_min=0, a_max=255) 

  for row in range(rows):
    for col in range(cols):
      img = imgs[count]
      ax[row,col].imshow(img.permute(1,2,0))
      ax[row,col].axis('off')
      count += 1

  fig.suptitle(name)
  fig.savefig("%s.png"%(name))
  if(show): plt.show()
  else: plt.close()
  
  
# Step 7: Train data with vae model, and periodically plot generated images to ensure they're becoming sharper
logger = SummaryWriter('logs/VAE')
epochs = 25
for epoch in range(epochs):
  RCL_Loss, KLD_Loss, Loss = 0, 0, 0
  total = 0
  for i, (x,_) in enumerate(ut_train_loader):
    x = x.to(device)
    optimizer.zero_grad()
    recon_x, mu, log_var = model_vae(x)
    loss, RCL, KLD = loss_func(recon_x, x, mu, log_var)
    loss.backward()
    optimizer.step()
    
    Loss += loss.item()
    RCL_Loss += RCL.item()
    KLD_Loss += KLD.item()   
    total += x.size(0)

  logger.add_scalar('Total Loss', Loss/total, epoch)
  logger.add_scalar('RCL Loss', RCL_Loss/total, epoch)
  logger.add_scalar('KLD Loss', KLD_Loss/total, epoch)

  generate_images(x, 3, 3, name="original image", show=True)
  generate_images(recon_x, 3, 3, name="generated image", show=True)
  print('Epoch: {} Loss: {:.4f} RCL: {:.4f} KLD: {:.4f}'.format(epoch+1, Loss/total, RCL_Loss/total, KLD_Loss/total))

%tensorboard  --logdir 'logs/VAE'


# Step 8: Randomly pick 5 images from the dataset, use the trained model to generate fake images
indices = np.random.choice(len(UT_dataset), 5, replace=False)
original_imgs = torch.empty([5,3,100,100], dtype=torch.float32)
reconstruct_imgs = torch.empty([5,3,100,100], dtype=torch.float32)
count = 0
for i in indices:
  original_imgs[count] = UT_dataset[i][0] 
  x = UT_dataset[i][0].unsqueeze(0).to(device)
  recon_x, mu, log_var = model_vae(x) 
  reconstruct_imgs[count] = recon_x.data.cpu()
  count += 1
  
# print original and generated images
plt.figure(figsize=(10,15))
for j in range(5):
  plt.subplot(5,2,2*j+1)
  plt.imshow(original_imgs[j].permute(1,2,0))
  plt.subplot(5,2,2*j+2)
  plt.imshow(reconstruct_imgs[j].permute(1,2,0))
plt.savefig('vae.png')


# Step 9: Randomly pick 2 images, use linear interpolation of their latent vector, to generate a video
for i, (x,_) in enumerate(ut_test_loader):
  if i == 1: break
  imgs = x

# pick 2 images (x and y) and compute their latent vectors
x, y = imgs[0].unsqueeze(0), imgs[1].unsqueeze(0)
x, y = x.to(device), y.to(device)
x, y = model_vae.encoder(x), model_vae.encoder(y)
x, y = x.view(x.size(0),-1), y.view(y.size(0),-1)
mu_x, log_var_x = model_vae.l1(x), model_vae.l2(x)
mu_y, log_var_y = model_vae.l1(y), model_vae.l2(y)
z_x, z_y = model_vae.reparametrize(mu_x, log_var_x), model_vae.reparametrize(mu_y, log_var_y)

# use linear interpolation to combine their latent vectors, and output a video that smoothly transforming from x to y
gif = []
for i in range(101):
  t = 0.01 * i
  z = t*z_x + (1-t)*z_y
  z = model_vae.l3(z)
  z = z.view(-1,6,8,8)
  recon_x = model_vae.decoder(z)
  img = recon_x.data.cpu()[0]
  img = transforms.ToPILImage()(img)
  gif.append(img)
gif[0].save('interpolation.gif', append_images=gif[1:], save_all=True, duration=200, loop=0)
