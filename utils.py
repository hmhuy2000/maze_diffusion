import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms 
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from PIL import Image


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

def forward_diffusion_sample(x_0, t, device="cpu"):
    """ 
    Takes an image and a timestep as input and 
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

# Define beta schedule
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

IMG_SIZE = 48
BATCH_SIZE = 512

class grid_dataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.len = 128 * 1000

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img = Image.open(f'test_figs/grid_{IMG_SIZE}.png')
        if self.transform:
            img = self.transform(img)
        return img

def load_transformed_dataset():
    data_transforms = [
        transforms.ToTensor(), # Scales data into [0,1] 
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
    ]
    data_transform = transforms.Compose(data_transforms)

    train = grid_dataset(root_dir=".",
                            transform=data_transform)
    test = grid_dataset(root_dir=".",
                            transform=data_transform)
    return torch.utils.data.ConcatDataset([train, test])

def show_tensor_image(image,t,num_show=0,pos=0,loss=None):
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
    if (t is not None and loss is not None):
        plt.title(f'x_{t.item()},loss={loss:.5f}',fontsize=60)
    plt.axis('off')
    plt.imshow(image,cmap='gray',vmin=0,vmax=255)

    image = image.point(lambda x: 200 if x>=180 else 0)
    plt.subplot(2, num_show, pos+num_show)
    plt.axis('off')
    plt.imshow(image,cmap='gray',vmin=0,vmax=255)

