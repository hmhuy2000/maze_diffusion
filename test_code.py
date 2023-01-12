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
from tqdm import trange


#-------------------------------------------------------#
from utils import *
from unet import *
#-------------------------------------------------------#

data = torchvision.datasets.StanfordCars(root=".", download=True)
data = load_transformed_dataset()
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
epochs = 1000 
log_file = open("logs.csv", "w")

model = SimpleUnet()
print("Num params: ", sum(p.numel() for p in model.parameters()))

#-------------------------------------------------------#

def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)

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
def sample_plot_image(epoch):
    # Sample noise
    img_size = IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=device)
    plt.figure(figsize=(15,15))
    plt.axis('off')
    num_images = 10
    stepsize = int(T/num_images)

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t)
        if i % stepsize == 0:
            plt.subplot(1, num_images, i//stepsize+1)
            show_tensor_image(img.detach().cpu())
    plt.savefig(f'figures/result_{epoch}.jpg')         

#-------------------------------------------------------#

model.to(device)
optimizer = Adam(model.parameters(), lr=0.001)

for epoch in trange(epochs):
    for step, batch in enumerate(dataloader):
      optimizer.zero_grad()

      t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
      loss = get_loss(model, batch[0], t)
      loss.backward()
      optimizer.step()

      if epoch%10 == 0 and step == 0:
        log_file.write(f'{epoch},{loss:.5f}\n')
        log_file.flush()        
        save_path = f'pretrained/{epoch}.pt'
        torch.save(model.state_dict(), save_path)
        model.load_state_dict(torch.load(save_path))
        sample_plot_image(epoch)

log_file.close()