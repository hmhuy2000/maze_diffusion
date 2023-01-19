import psutil
print(f'Number of CPUs: {psutil.cpu_count()}')
p = psutil.Process()
arr_cpus = [i for i in range(50,60)]
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
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, up=False,prev_down=False,prev_up=False):
        super().__init__()
        self.text_emb_out_dim = 384   #TODO
        if up:
            intput_dim = in_ch
            if (not prev_up):
                intput_dim *= 3
            else:
                intput_dim *= 2
            self.conv1 = nn.Conv2d(intput_dim, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            intput_dim = in_ch
            if (not prev_down):
                intput_dim *= 2
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


class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self,device):
        super().__init__()
        self.device = device
        image_channels = 1
        down_channels = (64,128,256,512,1024)
        up_channels = (1024,512,256,128, 64)
        out_dim = 1

        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)
        self.prev_conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        self.prev_downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], \
                                    prev_down=True) \
                    for i in range(len(down_channels)-1)])

        self.prev_ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                 up=True,prev_up=True) \
            for i in range(len(up_channels)-1)])

        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1]) \
                    for i in range(len(down_channels)-1)])
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], up=True) \
                    for i in range(len(up_channels)-1)])

        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)
        self.last_ln = nn.Linear(2304, 384)
        self.relu = nn.ReLU()

    def forward(self,prev, x):
        # Initial conv
        x = self.conv0(x)
        prev = self.prev_conv0(prev)

        # Unet
        residual_inputs = []
        prev_residual_inputs = []
        for (down,prev_down) in zip(self.downs,self.prev_downs):
            x = torch.cat((x, prev), dim=1)   
            x = down(x)
            prev = prev_down(prev)
            residual_inputs.append(x)
            prev_residual_inputs.append(prev)


        for (up,prev_up) in zip(self.ups,self.prev_ups):
            residual_x = residual_inputs.pop()
            prev_residual_x = prev_residual_inputs.pop()
            x = torch.cat((x, residual_x,prev), dim=1)           
            x = up(x)
            prev = torch.cat((prev,prev_residual_x),dim=1)
            prev = prev_up(prev)

        x = self.output(x).flatten(1)
        x = self.last_ln(self.relu(x))
        return x

class total_model(nn.Module):
    def __init__(self,device):
        super().__init__()
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.image_model = SimpleUnet(device=device)
        print(sum(p.numel() for p in self.text_model.parameters() if p.requires_grad))
        print(sum(p.numel() for p in self.image_model.parameters() if p.requires_grad))

#------------------------------------------------------------------#

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs = 1000
    exp_name = '2_step_setence_embedding'
    figure_path = './figures'
    pretrained_path = './pretrained'
    log_path = './logs'
    create_dir(f'{figure_path}/{exp_name}')
    create_dir(f'{pretrained_path}/{exp_name}')
    data_train = load_transformed_dataset(root_dir='./dataset/2_step_train',csv_file='./dataset/dataset_2_step_train.csv')
    data_valid = load_transformed_dataset(root_dir='./dataset/2_step_test',csv_file='./dataset/dataset_2_step_test.csv')
    print(f'train dataset have {data_train.__len__()} samples')
    print(f'valid dataset have {data_valid.__len__()} samples')
    model = total_model(device=device)
    model.to(device)
    model.train()
    optimizer = Adam(model.parameters(), lr=0.0001)
    train_dataloader = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(data_valid, batch_size=BATCH_SIZE, drop_last=True)
    log_file = open(f'{log_path}/{exp_name}_logs.csv','a')

    best_loss = np.inf

    for epoch in range(num_epochs):
        pbar = tqdm(enumerate(train_dataloader))
        for step, batch in pbar:
            optimizer.zero_grad()

            prevs,promts,images,text = batch
            prevs = prevs.cuda()
            images = images.cuda()
            num_data = len(promts)

            text_embedding = model.text_model.get_embedding(promts,device)
            images_embedding = model.image_model(prevs,images)

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
            l1_loss = F.l1_loss(text_embedding, images_embedding)

            loss = (
                mse_loss+
                cosine_loss
            )
            loss.backward()
            optimizer.step()
            if (step %10 == 0):
                pbar.set_description(f'[train] epoch = {epoch}, train params = {sum(p.numel() for p in model.parameters() if p.requires_grad)},'+
            f'cosine_loss = {cosine_loss.item():.7f}, l1_loss = {l1_loss.item():.7f}, mse_loss = {mse_loss.item():.7f}, total_loss = {loss.item():.7f}')

        total_eval = []
        total_cosine = []
        total_l1 = []
        total_mse = []
        model.eval()
        log = None
        with torch.no_grad():
            pbar = tqdm(enumerate(valid_dataloader))
            for step, batch in pbar:
                prevs,promts,images,text = batch
                prevs = prevs.cuda()
                images = images.cuda()
                num_data = len(promts)
                text_embedding = model.text_model.get_embedding(promts,device)
                images_embedding = model.image_model(prevs,images)

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
                l1_loss = F.l1_loss(text_embedding, images_embedding)

                loss = (
                    mse_loss+
                    cosine_loss
                )
                total_eval.append(loss.item())
                total_l1.append(l1_loss.item())
                total_mse.append(mse_loss.item())
                total_cosine.append(cosine_loss.item())
                log = f'[valid] epoch = {epoch} cosine_loss = {np.mean(total_cosine):.7f}, l1_loss = {np.mean(total_l1):.7f}, mse_loss = {np.mean(total_mse):.7f}, total_loss = {np.mean(total_eval):.7f}'
                if (step%10 == 0):
                    pbar.set_description(log)
            
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