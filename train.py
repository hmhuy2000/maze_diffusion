import psutil
print(f'Number of CPUs: {psutil.cpu_count()}')
p = psutil.Process()

arr_cpus = [i for i in range(50,60)]
exp_name = '3_step_guidance'
figure_path = f'figures/{exp_name}'
pretrained_path = f'pretrained/{exp_name}'

p.cpu_affinity(arr_cpus)
print(f'CPU pool after assignment ({len(arr_cpus)}): {p.cpu_affinity()}')
import warnings
warnings.filterwarnings("ignore")

#-------------------------------------------------------#

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

#-------------------------------------------------------#
import torch
import torchvision
import matplotlib.pyplot as plt
from torch.optim import Adam
from tqdm import trange,tqdm
import seaborn as sns
from torch.optim.lr_scheduler import CosineAnnealingLR


def create_dir(name):
    try:
        os.mkdir(name)
    except:
        print(f'warning: {name} existed!')
#-------------------------------------------------------#
from utils import *
from unet import *
#-------------------------------------------------------#

data_train = load_transformed_dataset(root_dir='./dataset/3_step_train',csv_file='./dataset/dataset_3_step_train.csv')
data_valid = load_transformed_dataset(root_dir='./dataset/3_step_test',csv_file='./dataset/dataset_3_step_test.csv')
print(f'train dataset have {data_train.__len__()} samples')
print(f'valid dataset have {data_valid.__len__()} samples')
dataloader = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
valid_dataloader = DataLoader(data_valid, batch_size=BATCH_SIZE, drop_last=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
epochs = 1000
log_file = open(f"logs/{exp_name}.csv", "a")
log_file.write('epoch,learning_rate,train_loss,valid_loss,valid_no_guidance_loss,guidance_lower,trainning_param\n')
create_dir(figure_path)
create_dir(pretrained_path)
#-------------------------------------------------------#
model = SimpleUnet(device=device)
print(f'total model parameter: {sum(p.numel() for p in model.parameters())}')
#-------------------------------------------------------#
sentence_pretrained = './pretrained/3_step_setence_embedding/valid_sentence_embedding_model.pt'
print(f'load sentence_embedding_model pretrained from {sentence_pretrained}')
model.text0.load_state_dict(torch.load(sentence_pretrained))
start_epoch = 0

# pretrained = './pretrained/3_step_guidance/53.pt'
# start_epoch = int(pretrained.split('/')[-1].split('.')[0])+1
# print(f'load pretrained from {pretrained}')
# model.load_state_dict(torch.load(pretrained))
#-------------------------------------------------------#

def get_loss(model,prev, promt, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(prev,promt,x_noisy,t)
    return F.l1_loss(noise, noise_pred)
    # return F.mse_loss(noise, noise_pred)

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

@torch.no_grad()
def sample_plot_image(epoch, prev,promt,image,text):
    # Sample noise
    img_size = IMG_SIZE
    for idx in range(5):
        img = torch.randn((1, 1, img_size, img_size), device=device)
        plt.figure(figsize=(100,20))
        plt.clf()
        num_images = 10
        stepsize = int(T/num_images)
        show_tensor_image(prev[idx].detach().cpu(),text[idx],image[idx].detach().cpu(),None,num_images+1, 1)

        for i in range(0,T)[::-1]:
            t = torch.full((1,), i, device=device, dtype=torch.long)
            img = sample_timestep(prev[idx].unsqueeze(dim=0),[promt[idx]],img, t)
            if i % stepsize == 0:
                show_tensor_image(None,text[idx],img.detach().cpu(),t,num_images+1, i//stepsize+2)
        plt.tight_layout()
        plt.savefig(f'{figure_path}/result_{epoch}/{idx}.png')         


#-------------------------------------------------------#
init_learning_rate = 0.001
model.to(device)
model.train()
optimizer = Adam(model.parameters(), lr=init_learning_rate)
print('freeze setence transformer')
for param in model.text0.parameters():
    param.requires_grad = False
best_val_loss = np.inf
for epoch in range(start_epoch,epochs):

    for g in optimizer.param_groups:
        g['lr'] = max(0.0001,init_learning_rate*(0.95**epoch))

    pbar = tqdm(enumerate(dataloader))
    if (epoch == 50):
        print('\nunfreeze setence transformer')
        for param in model.text0.parameters():
            param.requires_grad = True
    total_train_loss = []
    for step, batch in pbar:
        prevs,promts,images,text = batch
        prevs = prevs.cuda()
        images = images.cuda()
        optimizer.zero_grad()
        
        t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
        loss = get_loss(model,prevs,promts, images, t)
        total_train_loss.append(loss.detach().item())
        loss.backward()
        optimizer.step()
        if (step %10 == 0):
            pbar.set_description(f'[Train] epoch = {epoch},num_param = {sum(p.numel() for p in model.parameters() if p.requires_grad)}, learning rate = {optimizer.param_groups[0]["lr"]:.5f}'+
            f', loss = {np.mean(total_train_loss):.7f}')

    pbar = tqdm(enumerate(valid_dataloader))
    total_valid_loss = []
    total_valid_guidance_free = []
    prevs,promts,images,text = None,None,None,None
    with torch.no_grad():
        for step, batch in pbar:
            prevs,promts,images,text = batch
            prevs = prevs.cuda()
            images = images.cuda()
            
            t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
            loss = get_loss(model,prevs,promts, images, t)
            guidance_free_loss = get_loss(model,prevs,['' for _ in range(len(promts))], images, t)
            total_valid_loss.append(loss.detach().item())
            total_valid_guidance_free.append(guidance_free_loss.detach().item())

            if (step == 0):
                guidance_embedding = model.text0.get_embedding(comd,device)
                cosine_table = np.zeros((len(comd),len(comd)))
                for i in range(len(comd)):
                    for j in range(len(comd)):
                        if (i == j):
                            cosine_table[i,j] = F.cosine_embedding_loss(guidance_embedding[i],guidance_embedding[j],torch.tensor(1).cuda())
                        else:
                            cosine_table[i,j] = F.cosine_embedding_loss(guidance_embedding[i],guidance_embedding[j],torch.tensor(-1).cuda())
                plt.figure(figsize=(20,20))
                plt.clf()
                heatmap = sns.heatmap(cosine_table,vmin=0.0,vmax=1.0,annot=True,fmt='.2f')
                create_dir(f'{figure_path}/result_{epoch}')
                heatmap.figure.savefig(f'{figure_path}/result_{epoch}/cosine_result.png')

            if (step %10 == 0):
                pbar.set_description(f'[Valid] epoch = {epoch}'+
                f', loss = {np.mean(total_valid_loss):.7f}, guidance-free loss = {np.mean(total_valid_guidance_free):.7f},'+
                f'guidance lower = {-np.mean(total_valid_loss)+np.mean(total_valid_guidance_free)}')

    log_file.write(f'{epoch},{optimizer.param_groups[0]["lr"]:.5f},{np.mean(total_train_loss):.7f},{np.mean(total_valid_loss):.7f},{np.mean(total_valid_guidance_free):.7f},'+
    f'{-np.mean(total_valid_loss)+np.mean(total_valid_guidance_free)},{sum(p.numel() for p in model.parameters() if p.requires_grad)}\n')
    log_file.flush()      
    if (np.mean(total_valid_loss)<best_val_loss):
        best_val_loss = np.mean(total_valid_loss)
        print(f'new best valid loss: {best_val_loss:.7f}')
        save_path = f'{pretrained_path}/{epoch}.pt'
        torch.save(model.state_dict(), save_path)
    sample_plot_image(epoch, prevs,promts,images,text)
    

log_file.close()