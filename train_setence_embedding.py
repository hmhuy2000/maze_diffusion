import psutil
print(f'Number of CPUs: {psutil.cpu_count()}')
p = psutil.Process()
arr_cpus = [i for i in range(80,96)]
p.cpu_affinity(arr_cpus)
print(f'CPU pool after assignment ({len(arr_cpus)}): {p.cpu_affinity()}')
import warnings
warnings.filterwarnings("ignore")

def create_dir(name):
    try:
        os.mkdir(name)
    except:
        print(f'warning: {name} existed!')
from torch import nn
from torch.optim import Adam
import math
import torch
from tqdm import trange,tqdm
import torchvision
from sentence_transformers import SentenceTransformer
from utils import *
import seaborn as sns
#------------------------------------------------------------------#

import wandb
wandb.init(project='maze_diffusion', settings=wandb.Settings(_disable_stats=True), \
        group='mask_prediction_sentence', name='0', entity='hmhuy')
#------------------------------------------------------------------#
reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

#------------------------------------------------------------------#
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, up=False,prev=False):
        super().__init__()
        self.text_emb_out_dim = 384
        if up:
            intput_dim = in_ch
            intput_dim *= 2
            self.conv1 = nn.Conv2d(intput_dim, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            intput_dim = in_ch
            self.conv1 = nn.Conv2d(intput_dim, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        
    def forward(self, x):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class Unet(nn.Module):
    def __init__(self,device):
        super().__init__()
        self.device = device
        image_channels = 1
        down_channels = (64,128,256,512,1024)
        up_channels = (1024,512,256,128, 64)
        out_dim = 1
        text_emb_dim = 384
        
        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)
        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], ) \
                    for i in range(len(down_channels)-1)])
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], up=True) \
                    for i in range(len(up_channels)-1)])
        
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)
        self.last_ln = nn.Linear(IMG_SIZE*IMG_SIZE, text_emb_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Initial conv
        x = self.conv0(x)

        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x)
            residual_inputs.append(x)

        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)           
            x = up(x)

        x = self.output(x).flatten(1)
        x = self.last_ln(self.relu(x))
        return x


class total_model(nn.Module):
    def __init__(self,device):
        super().__init__()
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.image_model = Unet(device=device)
        print(sum(p.numel() for p in self.text_model.parameters() if p.requires_grad))
        print(sum(p.numel() for p in self.image_model.parameters() if p.requires_grad))

#------------------------------------------------------------------#

def main():
    train_root_dir = './dataset/easy_region_train'
    valid_root_dir = './dataset/easy_region_test'
    train_csv = './dataset/dataset_easy_region_train.csv'
    valid_csv = './dataset/dataset_easy_region_test.csv'
    comd = []
    for line in tqdm(open(train_csv,'r').readlines()):
        comd.append(line.split(',')[1])
    comd = list(set(comd))
    comd.sort()
    print('list promts = ',comd)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs = 1000
    exp_name = 'mask_prediction_sentence'
    figure_path = './figures'
    pretrained_path = './pretrained'
    log_path = './logs'
    create_dir(f'{figure_path}/{exp_name}')
    create_dir(f'{pretrained_path}/{exp_name}')
    data_train = load_transformed_dataset(root_dir=train_root_dir,csv_file=train_csv)
    # data_train = load_transformed_dataset(root_dir=valid_root_dir,csv_file=valid_csv)
    data_valid = load_transformed_dataset(root_dir=valid_root_dir,csv_file=valid_csv)
    print(f'train dataset have {data_train.__len__()} samples')
    print(f'valid dataset have {data_valid.__len__()} samples')
    model = total_model(device=device)
    model.to(device)
    model.train()

    sentence_pretrained = './pretrained/mask_prediction_sentence/valid_sentence_embedding_model.pt'
    print(f'load sentence_embedding_model pretrained from {sentence_pretrained}')
    model.text_model.load_state_dict(torch.load(sentence_pretrained))
    image_pretrained = './pretrained/mask_prediction_sentence/valid_image_embedding_model.pt'
    print(f'load image_embedding_model pretrained from {image_pretrained}')
    model.image_model.load_state_dict(torch.load(image_pretrained))

    optimizer = Adam(model.parameters(), lr=0.0001)
    train_dataloader = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(data_valid, batch_size=BATCH_SIZE, drop_last=True)
    log_file = open(f'{log_path}/{exp_name}_logs.csv','a')

    best_loss = np.inf
    train_loss = []
    train_cosine_loss = []
    train_mse_loss = []
    for epoch in range(num_epochs):
        pbar = tqdm(enumerate(train_dataloader), total=data_train.__len__()//BATCH_SIZE)
        for step, batch in pbar:
            optimizer.zero_grad()
            prevs,promts,images,Xmins,Ymins,Xmaxs,Ymaxs = batch
            if (step%100 == 0):
                reverse_transforms(prevs[0].cpu()).save('tmp/prev.png')
                reverse_transforms(images[0].cpu()).save('tmp/image.png')
                reverse_transforms((images[0]-prevs[0]).cpu()).save('tmp/target.png')
            images = (images-prevs).cuda()
            num_data = len(promts)

            text_embedding = model.text_model.get_embedding(promts,device)
            images_embedding = model.image_model(images)
            input1 = []
            input2 = []
            labels = []
            for i in range(num_data):
                for j in range(i,num_data):
                    input1.append(images_embedding[i])
                    input2.append(images_embedding[j])
                    if (promts[i] == promts[j]):
                        labels.append(1)
                    else:
                        labels.append(-1)

            labels = torch.tensor(labels).cuda()
            input1 = torch.stack(input1).cuda()
            input2 = torch.stack(input2).cuda()

            cosine_loss = F.cosine_embedding_loss(input1,input2,labels)
            mse_loss = F.mse_loss(text_embedding, images_embedding)

            loss = (
                mse_loss+
                cosine_loss
            )
            train_loss.append(loss.item())
            train_cosine_loss.append(cosine_loss.item())
            train_mse_loss.append(mse_loss.item())
            loss.backward()
            optimizer.step()
            if (step %10 == 0):
                pbar.set_description(f'[train] epoch = {epoch}, train params = {sum(p.numel() for p in model.parameters() if p.requires_grad)},'+
            f'cosine_loss = {np.mean(train_cosine_loss):.7f}, mse_loss = {np.mean(train_mse_loss):.7f}, total_loss = {np.mean(train_loss):.7f}')

        total_eval = []
        total_cosine = []
        total_mse = []
        model.eval()
        log = None
        with torch.no_grad():
            pbar = tqdm(enumerate(valid_dataloader), total=data_valid.__len__()//BATCH_SIZE)
            for step, batch in pbar:
                prevs,promts,images,Xmins,Ymins,Xmaxs,Ymaxs = batch
                images = (images-prevs).cuda()
                num_data = len(promts)

                text_embedding = model.text_model.get_embedding(promts,device)
                images_embedding = model.image_model(images)

                input1 = []
                input2 = []
                labels = []
                for i in range(num_data):
                    for j in range(i,num_data):
                        input1.append(images_embedding[i])
                        input2.append(images_embedding[j])
                        if (promts[i] == promts[j]):
                            labels.append(1)
                        else:
                            labels.append(-1)

                labels = torch.tensor(labels).cuda()
                input1 = torch.stack(input1).cuda()
                input2 = torch.stack(input2).cuda()

                cosine_loss = F.cosine_embedding_loss(input1,input2,labels)
                mse_loss = F.mse_loss(text_embedding, images_embedding)

                loss = (
                    mse_loss+
                    cosine_loss
                )
                total_eval.append(loss.item())
                total_mse.append(mse_loss.item())
                total_cosine.append(cosine_loss.item())
                log = f'[valid] epoch = {epoch} cosine_loss = {np.mean(total_cosine):.7f}, mse_loss = {np.mean(total_mse):.7f}, total_loss = {np.mean(total_eval):.7f}'
                if (step%10 == 0):
                    pbar.set_description(log)
            wandb.log({
                'sentence_loss/train_loss':np.mean(train_loss),
                'sentence_loss/train_mse_loss':np.mean(train_mse_loss),
                'sentence_loss/train_cosine_loss':np.mean(train_cosine_loss),
                'sentence_loss/valid_loss':np.mean(total_eval),
                'sentence_loss/valid_mse_loss':np.mean(total_mse),
                'sentence_loss/valid_cosine_loss':np.mean(total_cosine),
                       },step = epoch)
            guidance_embedding = model.text_model.get_embedding(comd,device)
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
            create_dir(f'{figure_path}/{exp_name}/result_{epoch}')
            heatmap.figure.savefig(f'{figure_path}/{exp_name}/result_{epoch}/cosine_result.png')
            wandb.log({"heatmap":wandb.Image(Image.open(f'{figure_path}/{exp_name}/result_{epoch}/cosine_result.png'))},step=epoch)
        
        log_file.write(f'{log}\n')
        log_file.flush()
        model.train()
        if (np.mean(total_eval)<best_loss):
            print(f'new best eval {np.mean(total_eval):.7f}')
            best_loss = np.mean(total_eval)
            torch.save(model.text_model.state_dict(), f'{pretrained_path}/{exp_name}/valid_sentence_embedding_model.pt')
            torch.save(model.image_model.state_dict(), f'{pretrained_path}/{exp_name}/valid_image_embedding_model.pt')
        else:
            print(f'eval = {np.mean(total_eval):.7f}, best eval = {best_loss:.7f}')
    log_file.close()

if __name__ == '__main__':
    main()