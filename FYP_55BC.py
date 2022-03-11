import os
import torch
import torchvision
import torch.nn as nn
import tensorflow.keras
from tqdm.notebook import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from IPython.display import Image
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.utils import save_image
from scipy.stats import norm
from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import add
from keras.utils.vis_utils import plot_model

from google.colab import drive
drive.mount('/content/drive')

sample_dir = "thermal"
DATA_DIR = '/content/dataset/'
norms = (.5, .5, .5), (.5, .5, .5)
latent_size=128
epochs = 1
lr_d = 10e-5
lr_g = 10e-4
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
fixed_latent = torch.randn(128, latent_size, 1, 1, device=device)

def denorm(img_tensor):
    return img_tensor * norms[1][0] + norms[0][0]

transform_low = T.Compose([
    T.Resize((128, 128)),T.CenterCrop(128), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
    T.ToTensor(),T.Normalize(*norms)
])
transform_mid = T.Compose([
    T.Resize((256, 256)),T.CenterCrop(256), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
    T.ToTensor(),T.Normalize(*norms)
])

transform_high = T.Compose([
    T.Resize((512, 512)),T.CenterCrop(512), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
    T.ToTensor(),T.Normalize(*norms)])

# def save_sample(index, fixed_latent, show=True):
#     fake_images = netG[0](fixed_latent)
#     fake_fname = "generated-images-{0:0=4d}.png".format(index)
#     save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=8)
#     if show:
#         fig, ax = plt.subplots(figsize=(8,8))
#         ax.set_xticks([]); ax.set_yticks([])
#         ax.imshow(make_grid(fake_images.cpu().detach()[:32], nrow=8).permute(1,2,0))
def show_image(train_dl):
    for images,_ in train_dl:
        fig, ax = plt.subplots(figsize=(8,8))
        ax.set_xticks([]); ax.set_yticks([])
        print(images.shape)
        ax.imshow(make_grid(denorm(images.detach()[:32]), nrow=8).permute(1,2,0))
        break

data = torchvision.datasets.ImageFolder(root=DATA_DIR, transform=transform_mid)
train_dl = DataLoader(data, 128, num_workers=2, pin_memory=True)
show_image(train_dl)

from keras import layers
class residual_block(nn.Module):

    def __init__(self,channel_in,channel_out,ksize,ssize):
        super(residual_block,self).__init__()
        self.upper = nn.Conv2d(channel_in, channel_out, kernel_size=ksize,stride=ssize,padding=1)
        self.lower = nn.Conv2d(channel_in, channel_out, kernel_size=ksize,stride=ssize,padding=1)
        self.batch_norm0 = nn.BatchNorm2d(channel_out)
        self.batch_norm1 = nn.BatchNorm2d(channel_out)
    def forward(self,x):
        upper = self.batch_norm0(self.upper(x))
        lower = self.batch_norm1(self.lower(x))
        return torch.add(upper,lower)

class Generator_zero(nn.Module):

    def __init__(self):
        super(Generator_zero, self).__init__()
        self.conv0 = nn.Conv2d(3, 16, kernel_size=7, stride=1)
        self.conv1 = nn.Conv2d(16, 64, kernel_size=7, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=7, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=7, stride=2)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=7, stride=2)
        self.convT0 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2)
        self.convT1 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2)
        self.convT2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2)
        self.convT3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2)
        self.convT4 = nn.Conv2d(32, 3, kernel_size=7, stride=1)
        self.batch_norm0 = nn.BatchNorm2d(64)
        self.batch_norm1 = nn.BatchNorm2d(128)
        self.batch_norm2 = nn.BatchNorm2d(256)
        self.batch_norm3 = nn.BatchNorm2d(256)
        self.batch_norm4 = nn.BatchNorm2d(128)
        self.batch_norm5 = nn.BatchNorm2d(128)
        self.batch_norm6 = nn.BatchNorm2d(64)
        self.batch_norm7 = nn.BatchNorm2d(32)
        self.layers = nn.ModuleList()
        for i in range(10):
            self.layers.append(residual_block(256,256,3,1))
    
    def forward(self, x,):
        x = self.conv0(x)
        x = F.relu(self.batch_norm0(self.conv1(x)))
        x = F.relu(self.batch_norm1(self.conv2(x)))
        x = F.relu(self.batch_norm2(self.conv3(x)))
        x = F.relu(self.batch_norm3(self.conv4(x)))
        for layer in self.layers[:-1]:
            x = layer(x)
        residual = x
        x = F.relu(self.batch_norm4(self.convT0(x)))
        x = F.relu(self.batch_norm5(self.convT1(x)))
        x = F.relu(self.batch_norm6(self.convT2(x)))
        x = F.relu(self.batch_norm7(self.convT3(x)))
        x = F.tanh(self.convT4(x))
        return x,residual

class Generator_one(nn.Module):

    def __init__(self):
        super(Generator_one, self).__init__()
        self.conv0 = nn.Conv2d(3, 64, kernel_size=10, stride=10,padding=1)
        self.conv1 = nn.Conv2d(64, 256, kernel_size=8, stride=8)
        self.convT0 = nn.ConvTranspose2d(256, 32, kernel_size=3, stride=3)
        self.convT1 = nn.Conv2d(32, 3, kernel_size=7, stride=1)
        self.batch_norm0 = nn.BatchNorm2d(256)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.layers = nn.ModuleList()
        for i in range(7):
            self.layers.append(residual_block(256,256,3,1))
    
    def forward(self, x,low_res):
        x = self.conv0(x)
        x = F.relu(self.batch_norm0(self.conv1(x)))
        print(x.shape,"x",low_res.shape)
        x = torch.add(x,low_res)
        for layer in self.layers[:-1]:
            x = layer(x)
        residual = x.clone().detach()
        x = F.relu(self.batch_norm1(self.convT0(x)))
        x = F.tanh(self.convT1(x))
        return x,residual

class Generator_two(nn.Module):

    def __init__(self):
        super(Generator_two, self).__init__()
        self.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1)
        self.convT0 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2)
        self.convT1 = nn.Conv2d(32, 3, kernel_size=7, stride=1)
        self.batch_norm0 = nn.BatchNorm2d(32)

        self.layers = nn.ModuleList()
        for i in range(4):
            self.layers.append(residual_block(64,64,3,1))
    
    def forward(self, x,mid_res):
        x = self.conv0(x)
        torch.add(x,mid_res)
        for layer in self.layers[:-1]:
            x = layer(x)
        residual = x.clone().detach()
        x = F.relu(self.batch_norm0(self.convT0(x)))
        x = F.tanh(self.convT1(x))
        return x

class Discriminator_zero(nn.Module):

    def __init__(self):
        super(Discriminator_zero, self).__init__()
        self.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1)
        self.conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(512, 1, kernel_size=3, stride=2)
        self.batch_norm0 = nn.BatchNorm2d(128)
        self.batch_norm1 = nn.BatchNorm2d(256)
        self.batch_norm2 = nn.BatchNorm2d(512)
        self.batch_norm3 = nn.BatchNorm2d(1)
    
    def forward(self, x,matching=False):
        x = self.conv0(x)
        x = F.relu(self.batch_norm0(self.conv1(x)))
        x = F.relu(self.batch_norm1(self.conv2(x)))
        x = F.relu(self.batch_norm2(self.conv3(x)))
        x = F.relu(self.batch_norm3(self.conv4(x)))

        return x

class Discriminator_one(nn.Module):

    def __init__(self):
        super(Discriminator_one, self).__init__()
        self.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1)
        self.conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(512, 1, kernel_size=3, stride=2,padding=1)
        self.batch_norm0 = nn.BatchNorm2d(128)
        self.batch_norm1 = nn.BatchNorm2d(256)
        self.batch_norm2 = nn.BatchNorm2d(512)
        self.batch_norm3 = nn.BatchNorm2d(1)
    
    def forward(self, x,matching=False):
        x = self.conv0(x)
        x = F.relu(self.batch_norm0(self.conv1(x)))
        x = F.relu(self.batch_norm1(self.conv2(x)))
        x = F.relu(self.batch_norm2(self.conv3(x)))
        x = F.relu(self.batch_norm3(self.conv4(x)))
        
        return x
class Discriminator_two(nn.Module):

    def __init__(self):
        super(Discriminator_two, self).__init__()
        self.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1)
        self.conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(512, 1, kernel_size=3, stride=2)
        self.batch_norm0 = nn.BatchNorm2d(128)
        self.batch_norm1 = nn.BatchNorm2d(256)
        self.batch_norm2 = nn.BatchNorm2d(512)
        self.batch_norm3 = nn.BatchNorm2d(1)
    
    def forward(self, x,matching=False):
        x = self.conv0(x)
        x = F.relu(self.batch_norm0(self.conv1(x)))
        x = F.relu(self.batch_norm1(self.conv2(x)))
        x = F.relu(self.batch_norm2(self.conv3(x)))
        x = F.relu(self.batch_norm3(self.conv4(x)))
        
        return x
