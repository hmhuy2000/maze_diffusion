import psutil
print(f'Number of CPUs: {psutil.cpu_count()}')
p = psutil.Process()

arr_cpus = [i for i in range(60,66)]

p.cpu_affinity(arr_cpus)
print(f'CPU pool after assignment ({len(arr_cpus)}): {p.cpu_affinity()}')
import warnings
warnings.filterwarnings("ignore")


comd = [
    'Add a room in the top left with size 2',
    'Add a room in the top left with size 3',
    'Add a room in the top left with size 4',

    'Add a room in the top right with size 2',
    'Add a room in the top right with size 3',
    'Add a room in the top right with size 4',

    'Add a room in the bottom left with size 2',
    'Add a room in the bottom left with size 3',
    'Add a room in the bottom left with size 4',

    'Add a room in the bottom right with size 2',
    'Add a room in the bottom right with size 3',
    'Add a room in the bottom right with size 4',
]

import torch
import torchvision
import matplotlib.pyplot as plt
from torch.optim import Adam
from tqdm import trange,tqdm
import seaborn as sns
from PIL import Image
from copy import deepcopy

from utils import *
from unet import *
device = "cuda" if torch.cuda.is_available() else "cpu"

model = SimpleUnet(device=device)
print(f'total model parameter: {sum(p.numel() for p in model.parameters())}')
pretrained = './pretrained/2_step_guidance/11.pt'
print(f'load pretrained from {pretrained}')
model.load_state_dict(torch.load(pretrained))
model.cuda()
model.eval()

@torch.no_grad()
def sample_timestep(prev,promt,x, t):
    """
    Calls the model to predict the noise in the image and returns 
    the denoised image. 
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
    
    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(prev,promt,x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
    
    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

root_dir = './dataset/2_step_test'
target_dir = './figures/test'
origin_image = np.asarray(Image.open(f'{root_dir}/0000000.png'))
grid_size = origin_image.shape[0]
new_input_image = np.zeros((grid_size*3,grid_size*3),dtype=np.uint8)
for x in range(grid_size):
    for y in range(grid_size):
        new_input_image[x*3:x*3+3,y*3:y*3+3] = origin_image[x,y]
origin_image = new_input_image
image = Image.fromarray(origin_image)
image.save(f'{target_dir}/origin.png')
transforms = transforms.Compose([
        transforms.ToTensor(), # Scales data into [0,1] 
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
    ])
stepsize=30
num_images = 10
list_promts = [
    'Add a room in the top right with size 3',
    'Add a room in the bottom left with size 4',
    'Add a room in the top left with size 3',
    'Add a room in the bottom right with size 4',
]
input_image = deepcopy(origin_image)
input_image = transforms(input_image).unsqueeze(0).cuda()

print(list_promts)

for loop in range(len(list_promts)):
    promt = list_promts[loop]
    print(f'process {promt}:')
    if (loop>0):
        input_image = deepcopy(img)

    img_size = input_image.shape[-1]
    img = torch.randn((1, 1, img_size, img_size), device=device)
    print(input_image.shape,img.shape)
    plt.figure(figsize=(100,20))
    plt.clf()

    for i in tqdm(range(0,T)[::-1]):
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(input_image,[promt],img, t)
        if i % stepsize == 0:
            show_tensor_image(None,promt,img.detach().cpu(),t,num_images+1, i//stepsize+1)
    plt.tight_layout()
    plt.savefig(f'{target_dir}/image_{loop+1}.png')  