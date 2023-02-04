import psutil
print(f'Number of CPUs: {psutil.cpu_count()}')
p = psutil.Process()

arr_cpus = [i for i in range(80,96)]

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
import seaborn as sns
from copy import deepcopy


def create_dir(name):
    try:
        os.mkdir(name)
    except:
        print(f'warning: {name} existed!')
#-------------------------------------------------------#
from utils import *
from unet import *
#-------------------------------------------------------#

reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

#-------------------------------------------------------#
exp_name = 'new_64'
figure_path = f'figures/{exp_name}'
pretrained_path = f'pretrained/{exp_name}'
# data_train = load_transformed_dataset(root_dir='./dataset/64_region_train',csv_file='./dataset/dataset_64_region_train.csv')
data_train = load_transformed_dataset(root_dir='./dataset/64_3step_region_train_small',csv_file='./dataset/dataset_64_3step_region_train_small.csv')
data_valid = load_transformed_dataset(root_dir='./dataset/64_3step_region_test',csv_file='./dataset/dataset_64_3step_region_test.csv')
print(f'train dataset have {data_train.__len__()} samples')
print(f'valid dataset have {data_valid.__len__()} samples')
dataloader = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
valid_dataloader = DataLoader(data_valid, batch_size=BATCH_SIZE,shuffle=True, drop_last=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
epochs = 1000

#-------------------------------------------------------#

# prev,promt,image,Xmin,Ymin,Xmax,Ymax = next(iter(dataloader))
# print(image.shape)
# plt.figure(figsize=(120,20))
# plt.axis('off')
# num_images = 10
# stepsize = int(T/num_images)

# def show_test_image(image):
#     reverse_transforms = transforms.Compose([
#         transforms.Lambda(lambda t: (t + 1) / 2),
#         transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
#         transforms.Lambda(lambda t: t * 255.),
#         transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
#         transforms.ToPILImage(),
#     ])

#     # Take first image of batch
#     if len(image.shape) == 4:
#         image = image[0, :, :, :] 
#     plt.imshow(reverse_transforms(image),cmap='gray',vmin=0,vmax=255)

# for idx in range(0, T, stepsize):
#     t = torch.Tensor([idx]).type(torch.int64)
#     plt.subplot(1, num_images+2, (idx//stepsize) + 1)
#     new_image = image
#     if (idx>0):
#         new_image, _ = forward_diffusion_sample(image, t,Xmin,Ymin,Xmax,Ymax,device)
#     print(idx,'\t',new_image.max().item(),new_image.min().item(),new_image.mean().item(),new_image.std().item())
#     show_test_image(new_image.cpu())

# t = torch.Tensor([T-1]).type(torch.int64)
# plt.subplot(1, num_images+2, (T//stepsize) + 1)
# new_image, _ = forward_diffusion_sample(image, t,Xmin,Ymin,Xmax,Ymax,device)
# print(T-1,'\t',new_image.max().item(),new_image.min().item(),new_image.mean().item(),new_image.std().item())
# show_test_image(new_image.cpu())

# plt.savefig('test_img.png')
# exit()

#-------------------------------------------------------#

log_file = open(f"logs/{exp_name}.csv", "a")
log_file.write('epoch,learning_rate,train_loss,valid_loss,valid_no_guidance_loss,guidance_lower,trainning_param\n')
create_dir(figure_path)
create_dir(pretrained_path)
#-------------------------------------------------------#
model = SimpleUnet(device=device)
print(f'total model parameter: {sum(p.numel() for p in model.parameters())}')
#-------------------------------------------------------#
# sentence_pretrained = './pretrained/2_step_setence_embedding/valid_sentence_embedding_model.pt'
# print(f'load sentence_embedding_model pretrained from {sentence_pretrained}')
# model.text0.load_state_dict(torch.load(sentence_pretrained))
start_epoch = 0

# pretrained = './pretrained/region_3step_64/44.pt'
# start_epoch = int(pretrained.split('/')[-1].split('.')[0])+1
# print(f'load pretrained from {pretrained}')
# model.load_state_dict(torch.load(pretrained))
#-------------------------------------------------------#

def get_loss(model, promt, x_0, t,Xmins,Ymins,Xmaxs,Ymaxs):
    x_noisy, noise = forward_diffusion_sample(x_0, t, Xmins, Ymins, Xmaxs, Ymaxs, device)
    noise_pred = model(promt,x_noisy,t)

    # reverse_transforms(prev[0].cpu()).save('tmp_prev.png')
    # reverse_transforms(x_0[0].cpu()).save('tmp_x_0.png')
    # reverse_transforms(x_noisy[0].cpu()).save('tmp_x_noisy.png')
    # reverse_transforms(noise[0].cpu()).save('tmp_noise.png')
    # reverse_transforms(noise_pred[0].detach().cpu()).save('tmp_noise_pred.png')
    # print()
    # print('t = ',t[0].item())
    # print('prev: ',prev[0].mean().item(),prev[0].max().item(),prev[0].min().item(),prev[0].std().item())
    # print('x_0: ',x_0[0].mean().item(),x_0[0].max().item(),x_0[0].min().item(),x_0[0].std().item())
    # print('x_noisy: ',x_noisy[0].mean().item(),x_noisy[0].max().item(),x_noisy[0].min().item(),x_noisy[0].std().item())
    # print('noise: ',noise[0].mean().item(),noise[0].max().item(),noise[0].min().item(),noise[0].std().item())
    # exit()

    return F.l1_loss(noise, noise_pred)
    # return F.mse_loss(noise, noise_pred)

@torch.no_grad()
def sample_timestep(promt,x, t,Xmin,Ymin,Xmax,Ymax):
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
        x - betas_t * model(promt,x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
    if t == 0:
        return model_mean
    else:
        noise = torch.zeros_like(x)
        noise[0,:,Xmin:Xmax+1,Ymin:Ymax+1] = torch.randn(noise[0,:,Xmin:Xmax+1,Ymin:Ymax+1].shape)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

@torch.no_grad()
def sample_plot_image(epoch, prev,promt,image,Xmins,Ymins,Xmaxs,Ymaxs):
    # Sample noise
    img_size = IMG_SIZE
    for idx in range(5):
        img = deepcopy(prev[idx])
        Xmin,Ymin,Xmax,Ymax = Xmins[idx],Ymins[idx],Xmaxs[idx],Ymaxs[idx]

        noise = torch.zeros_like(img)
        noise[:,Xmin:Xmax+1,Ymin:Ymax+1] = torch.randn(noise[:,Xmin:Xmax+1,Ymin:Ymax+1].shape)

        sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, torch.full((1,), T-1, device=device, dtype=torch.long), img.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            sqrt_one_minus_alphas_cumprod, torch.full((1,), T-1, device=device, dtype=torch.long), img.shape
        )
        img = sqrt_alphas_cumprod_t.to(device) * img.to(device) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)

        # img[:,Xmin:Xmax+1,Ymin:Ymax+1] = torch.randn(img[:,Xmin:Xmax+1,Ymin:Ymax+1].shape)

        img = img.unsqueeze(0).cuda()

        print(f'{idx}, x_{T}',img.mean().item(),img.max().item(),img.min().item(),img.std().item())
        plt.figure(figsize=(120,20))
        plt.clf()
        num_images = 10
        stepsize = int(T/num_images)
        show_tensor_image(prev[idx].detach().cpu(),promt[idx],image[idx].detach().cpu(),None,num_images+2, 1)
        show_tensor_image(None,promt[idx],img.detach().cpu(),torch.full((1,), T, device=device, dtype=torch.long),num_images+2, T//stepsize+2)

        for i in range(0,T)[::-1]:
            t = torch.full((1,), i, device=device, dtype=torch.long)
            img = sample_timestep([promt[idx]],img, t,Xmin,Ymin,Xmax,Ymax)
            if i % stepsize == 0:
                show_tensor_image(None,promt[idx],img.detach().cpu(),t,num_images+2, i//stepsize+2)
        print(f'{idx}, x_{0}',img.mean().item(),img.max().item(),img.min().item(),img.std().item())
        plt.tight_layout()
        plt.savefig(f'{figure_path}/result_{epoch}/{idx}.png')         


#-------------------------------------------------------#
init_learning_rate = 0.001
model.to(device)
optimizer = Adam(model.parameters(), lr=init_learning_rate)
print('freeze setence transformer')
for param in model.text0.parameters():
    param.requires_grad = False
best_val_loss = np.inf
for epoch in range(start_epoch,epochs):

    for g in optimizer.param_groups:
        g['lr'] = max(0.0001,init_learning_rate*(0.95**epoch))
    model.train()
    pbar = tqdm(enumerate(dataloader))
    if (epoch == 5):
        print('\nunfreeze setence transformer')
        for param in model.text0.parameters():
            param.requires_grad = True
    total_train_loss = []
    for step, batch in pbar:
        prevs,promts,images,Xmins,Ymins,Xmaxs,Ymaxs = batch
        images = images.cuda()
        optimizer.zero_grad()
        
        t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
        loss = get_loss(model,promts, images, t,Xmins,Ymins,Xmaxs,Ymaxs)
        total_train_loss.append(loss.detach().item())
        loss.backward()
        optimizer.step()
        if (step %10 == 0):
            pbar.set_description(f'[Train] epoch = {epoch},num_param = {sum(p.numel() for p in model.parameters() if p.requires_grad)}, learning rate = {optimizer.param_groups[0]["lr"]:.5f}'+
            f', loss = {np.mean(total_train_loss):.7f}')

    model.eval()
    pbar = tqdm(enumerate(valid_dataloader))
    total_valid_loss = []
    total_valid_guidance_free = []
    prevs,promts,images,Xmins,Ymins,Xmaxs,Ymaxs = None,None,None,None,None,None,None
    with torch.no_grad():
        for step, batch in pbar:
            prevs,promts,images,Xmins,Ymins,Xmaxs,Ymaxs = batch
            images = images.cuda()
            t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
            loss = get_loss(model,promts, images, t,Xmins,Ymins,Xmaxs,Ymaxs)
            guidance_free_loss = get_loss(model,['' for _ in range(len(promts))], images, t,Xmins,Ymins,Xmaxs,Ymaxs)
            total_valid_loss.append(loss.detach().item())
            total_valid_guidance_free.append(guidance_free_loss.detach().item())

            if (step %10 == 0):
                pbar.set_description(f'[Valid] epoch = {epoch}'+
                f', loss = {np.mean(total_valid_loss):.7f}, guidance-free loss = {np.mean(total_valid_guidance_free):.7f},'+
                f'guidance lower = {-np.mean(total_valid_loss)+np.mean(total_valid_guidance_free)}')

    log_file.write(f'{epoch},{optimizer.param_groups[0]["lr"]:.5f},{np.mean(total_train_loss):.7f},{np.mean(total_valid_loss):.7f},{np.mean(total_valid_guidance_free):.7f},'+
    f'{-np.mean(total_valid_loss)+np.mean(total_valid_guidance_free):.7f},{sum(p.numel() for p in model.parameters() if p.requires_grad)}\n')
    log_file.flush()      
    if (np.mean(total_valid_loss)<best_val_loss):
        best_val_loss = np.mean(total_valid_loss)
        print(f'new best valid loss: {best_val_loss:.7f}')
        save_path = f'{pretrained_path}/{epoch}.pt'
        torch.save(model.state_dict(), save_path)
    create_dir(f'{figure_path}/result_{epoch}')
    sample_plot_image(epoch,prevs,promts,images,Xmins,Ymins,Xmaxs,Ymaxs)
    

log_file.close()