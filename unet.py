from torch import nn
import math
import torch
import torchvision
from sentence_transformers import SentenceTransformer

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None,text_emb_dim=None, up=False,prev=False):
        super().__init__()
        self.text_emb_dim = text_emb_dim
        self.time_emb_dim = time_emb_dim
        self.text_emb_out_dim = 384
        if (time_emb_dim):
            self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if (text_emb_dim):
            self.text_mlp =  nn.Linear(text_emb_dim, self.text_emb_out_dim) 

        if up:
            if (not prev):
                intput_dim = in_ch * 3
            else:
                intput_dim = in_ch * 2
            if (self.text_emb_dim):
                intput_dim += self.text_emb_out_dim
            self.conv1 = nn.Conv2d(intput_dim, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            intput_dim = in_ch
            if (not prev):
                intput_dim *= 2
            if (self.text_emb_dim):
                intput_dim += self.text_emb_out_dim
            self.conv1 = nn.Conv2d(intput_dim, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        
    def forward(self, x, t=None, text=None):
        if (self.text_emb_dim is None):
            assert text is None
        if (self.time_emb_dim is None):
            assert t is None

        if (self.text_emb_dim): #TODO
            assert text is not None
            text = self.relu(self.text_mlp(text))
            text = text[(..., ) + (None, ) * 2]
            text = text.repeat(1, 1, x.shape[2], x.shape[3])
            x = torch.cat((x,text),dim=1)

        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        
        # Time embedding
        if (self.time_emb_dim):
            assert t is not None
            time_emb = self.relu(self.time_mlp(t))
            time_emb = time_emb[(..., ) + (None, ) * 2]
            h = h + time_emb

        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        # embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        embeddings = torch.cat((embeddings.cos(),embeddings.sin()), dim=-1)
        return embeddings


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
        time_emb_dim = 32
        text_emb_dim = 384

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )
        
        self.text0 = SentenceTransformer('all-MiniLM-L6-v2')

        #----------------------------------------------------------#

        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], \
                                    time_emb_dim=time_emb_dim,text_emb_dim=text_emb_dim) \
                    for i in range(len(down_channels)-1)])
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                        time_emb_dim=time_emb_dim,text_emb_dim=text_emb_dim, up=True) \
                    for i in range(len(up_channels)-1)])
        #----------------------------------------------------------#

        # Initial projection
        self.prev_conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.prev_downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1],prev=True) \
                    for i in range(len(down_channels)-1)])
        # Upsample
        self.prev_ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], up=True,prev=True) \
                    for i in range(len(up_channels)-1)])
        #----------------------------------------------------------#

        self.output = nn.Conv2d(up_channels[-1]*2, out_dim, 1)

    def forward(self,prev, prompt, x, timestep):
        # Embedd time
        timestep = self.time_mlp(timestep)
        prompt = self.text0.get_embedding(prompt,self.device)
        # Initial conv
        x = self.conv0(x)
        prev = self.prev_conv0(prev)

        # Unet
        residual_inputs = []
        prev_residual_inputs = []
        for (prev_down,down) in zip(self.prev_downs,self.downs):
            x = torch.cat((x, prev), dim=1)  
            x = down(x, timestep,prompt)
            prev = prev_down(prev)
            residual_inputs.append(x)
            prev_residual_inputs.append(prev)

        for (prev_up,up) in zip(self.prev_ups,self.ups):
            residual_x = residual_inputs.pop()
            prev_residual = prev_residual_inputs.pop()
            x = torch.cat((x, residual_x,prev), dim=1)           
            x = up(x, timestep,prompt)
            prev = prev_up(torch.cat((prev,prev_residual), dim=1))

        x = self.output(torch.cat((x,prev), dim=1))
        return x

