import psutil
print(f'Number of CPUs: {psutil.cpu_count()}')
p = psutil.Process()
arr_cpus = [i for i in range(50,60)]
p.cpu_affinity(arr_cpus)
print(f'CPU pool after assignment ({len(arr_cpus)}): {p.cpu_affinity()}')
import warnings
warnings.filterwarnings("ignore")
#-------------------------------------------------------#
import torch
import torchvision
import matplotlib.pyplot as plt
from torch.optim import Adam
from tqdm import trange,tqdm


#-------------------------------------------------------#
from utils import *
from unet import *
#-------------------------------------------------------#

data = load_transformed_dataset()
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
epochs = 1000 
log_file = open("logs.csv", "w")
#-------------------------------------------------------#
# Simulate forward diffusion

# image = next(iter(dataloader))[0]

# plt.figure(figsize=(100,20))
# plt.axis('off')
# num_images = 10
# stepsize = int(T/num_images)

# for idx in range(0, T, stepsize):
#     t = torch.Tensor([idx]).type(torch.int64)
#     # plt.subplot(1, num_images+1, (idx//stepsize) + 1)
#     image, noise = forward_diffusion_sample(image, t)
#     show_tensor_image(image,t,num_show=num_images,pos=idx//stepsize+1)
#     plt.title(f'x_{t.item()}')

# plt.savefig('figures/show_tensor_image.jpg')
# exit()
#-------------------------------------------------------#
model = SimpleUnet()
print("Num params: ", sum(p.numel() for p in model.parameters()))

#-------------------------------------------------------#

def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)
    # return F.mse_loss(noise, noise_pred)

@torch.no_grad()
def sample_timestep(x, t):
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
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
    
    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

@torch.no_grad()
def sample_plot_image(epoch, batch):
    # Sample noise
    img_size = IMG_SIZE
    img = torch.randn((1, 1, img_size, img_size), device=device)
    plt.figure(figsize=(100,20))
    num_images = 10
    stepsize = int(T/num_images)
    show_tensor_image(batch[0].detach().cpu(),None,num_images+1, 1)

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t)
        mse_loss = F.mse_loss(batch[0],img.detach().cpu())
        if i % stepsize == 0:
            show_tensor_image(img.detach().cpu(),t,num_images+1, i//stepsize+2,mse_loss)
    plt.tight_layout()
    plt.savefig(f'figures/result_{epoch}.png')         

#-------------------------------------------------------#

model.to(device)
optimizer = Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
    for step, batch in tqdm(enumerate(dataloader)):
      optimizer.zero_grad()
      
      t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
      loss = get_loss(model, batch, t)
      loss.backward()
      optimizer.step()
      if (step == 0):
        print(f'epoch {epoch}:',loss.item())

      if epoch%1 == 0 and step == 0:
        log_file.write(f'{epoch},{loss:.5f}\n')
        log_file.flush()        
        save_path = f'pretrained/{epoch}.pt'
        torch.save(model.state_dict(), save_path)
        model.load_state_dict(torch.load(save_path))
        sample_plot_image(epoch, batch)
    

log_file.close()