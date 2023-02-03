import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms 
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from PIL import Image

import cv2

def show_images(datset, num_samples=20, cols=4):
    """ Plots some samples from the dataset """
    plt.figure(figsize=(15,15)) 
    for i, img in enumerate(datset):
        if i == num_samples:
            break
        plt.subplot(num_samples//cols + 1, cols, i + 1)
        plt.imshow(np.asarray(img))
    plt.savefig('figures/show_images.jpg')

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    """ 
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0,t,Xmins,Ymins,Xmaxs,Ymaxs, device="cpu"):
    """ 
    Takes an image and a timestep as input and 
    returns the noisy version of it
    """

    sample_size = len(Xmins)
    noise = torch.zeros_like(x_0)
    for idx in range(sample_size):
        Xmin,Ymin,Xmax,Ymax = Xmins[idx],Ymins[idx],Xmaxs[idx],Ymaxs[idx]
        noise[idx,:,Xmin:Xmax+1,Ymin:Ymax+1] = torch.randn(noise[idx,:,Xmin:Xmax+1,Ymin:Ymax+1].shape)

    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

# Define beta schedule
# T = 350
# betas = linear_beta_schedule(timesteps=T,start=1e-4,end=1e-2)
T = 300
betas = linear_beta_schedule(timesteps=T,start=1e-3,end=3e-2)
# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

IMG_SIZE = 64
scale_img = 3
IMG_SIZE = IMG_SIZE*scale_img
BATCH_SIZE = 64

class grid_dataset(Dataset):

    def __init__(self, root_dir,csv_file, transform=None):
        self.root_dir = root_dir
        self.csv_file = open(csv_file,'r').readlines()
        self.transform = transform

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):

        line = self.csv_file[idx].replace('\n','').split(',')

        input_image = np.asarray(Image.open(f'{self.root_dir}/{line[0]}'))
        promt = line[1]
        Xmin,Ymin,Xmax,Ymax = int(line[2])*scale_img,int(line[3])*scale_img,int(line[4])*scale_img,int(line[5])*scale_img
        output_image = np.asarray(Image.open(f'{self.root_dir}/{line[-1]}'))
        assert Xmin<Xmax and Ymin<Ymax

        new_input_image = cv2.resize(input_image, (IMG_SIZE,IMG_SIZE), interpolation = cv2.INTER_AREA)
        new_out_image = cv2.resize(output_image, (IMG_SIZE,IMG_SIZE), interpolation = cv2.INTER_AREA)

        if self.transform:
            new_input_image = self.transform(new_input_image)
            new_out_image = self.transform(new_out_image)
        
        return (new_input_image,promt,new_out_image,Xmin,Ymin,Xmax,Ymax)

def load_transformed_dataset(root_dir,csv_file):
    data_transforms = [
        transforms.ToTensor(), # Scales data into [0,1] 
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
    ]
    data_transform = transforms.Compose(data_transforms)

    train = grid_dataset(root_dir=root_dir,csv_file=csv_file,
                            transform=data_transform)
    return train

def show_tensor_image(prev,promt,image,t,num_show=0,pos=0):
    image = image.clamp(min=-1,max=1)
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])
    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    image = reverse_transforms(image)

    plt.subplot(2, num_show, pos)
    if (prev is not None):
        prev = reverse_transforms(prev)
        plt.axis('off')
        plt.imshow(prev,cmap='gray',vmin=0,vmax=255)
        plt.title(f'previous state',fontsize=60)

        plt.subplot(2, num_show, pos+num_show)
        plt.imshow(image,cmap='gray',vmin=0,vmax=255)
        plt.title(f'promt: {promt}',fontsize=30)
        plt.axis('off')

    else:
        plt.title(f'x_{t.item()}',fontsize=50)
        plt.imshow(image,cmap='gray',vmin=0,vmax=255)
        plt.axis('off')
