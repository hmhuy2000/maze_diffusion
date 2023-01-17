from torch import nn
import math
import torch
import torchvision
from sentence_transformers import SentenceTransformer

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim,text_emb_dim=None, up=False,prev_up=False):
        super().__init__()
        self.text_emb_dim = text_emb_dim
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if (text_emb_dim):
            self.text_mlp =  nn.Linear(text_emb_dim, 32)
        if up:
            intput_dim = 3*in_ch
            if (text_emb_dim):
                intput_dim += 32
            self.conv1 = nn.Conv2d(intput_dim, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            intput_dim = in_ch
            if (prev_up==False):
                intput_dim *= 2
            if (text_emb_dim):
                intput_dim += 32
            self.conv1 = nn.Conv2d(intput_dim, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        
    def forward(self, x, t, text=None):
        if (self.text_emb_dim):
            text = self.relu(self.text_mlp(text))
            text = text[(..., ) + (None, ) * 2]
            text = text.expand(-1,-1,x.shape[2],x.shape[3])
            x = torch.cat((x,text),dim=1)

        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        
        # Time embedding
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

        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)
        self.prev_conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.prev_downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], \
                                    time_emb_dim,text_emb_dim,prev_up=True) \
                    for i in range(len(down_channels)-1)])

        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], \
                                    time_emb_dim,text_emb_dim) \
                    for i in range(len(down_channels)-1)])
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                        time_emb_dim,text_emb_dim, up=True) \
                    for i in range(len(up_channels)-1)])

        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self,prev, promt, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)

        # promt = self.text0.encode(promt,convert_to_tensor=True,device=self.device)
        promt = self.text0.get_embedding(promt,self.device)
        # Initial conv
        x = self.conv0(x)
        prev = self.prev_conv0(prev)

        # Unet
        residual_inputs = []
        prev_residual_inputs = []
        for (down,prev_down) in zip(self.downs,self.prev_downs):
            x = torch.cat((x, prev), dim=1)   
            x = down(x, t,promt)
            prev = prev_down(prev,t,promt)

            residual_inputs.append(x)
            prev_residual_inputs.append(prev)

        for up in self.ups:
            residual_x = residual_inputs.pop()
            prev_residual_x = prev_residual_inputs.pop()

            x = torch.cat((x, residual_x,prev_residual_x), dim=1)           
            x = up(x, t,promt)

        x = self.output(x)
        return x

