import torch
from x_clip import CLIP
from x_clip.tokenizer import SimpleTokenizer
import numpy as np
from tqdm import trange
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = SimpleTokenizer()
sentence = ['this is a dog','this is a cat','this is a mouse']
clip = CLIP(
    dim_text = 384,
    dim_image = 64,
    dim_latent = 256,
    num_text_tokens = 10000,
    text_enc_depth = 6,
    text_seq_len = 64,
    text_heads = 8,
    visual_enc_depth = 6,
    visual_image_size = 256,
    visual_patch_size = 32,
    visual_heads = 8,
    visual_patch_dropout = 0.5,             # patch dropout probability, used in Kaiming He's FLIP to save compute and improve end results - 0.5 is good value, 0.75 on high end is tolerable
    use_all_token_embeds = False,           # whether to use fine-grained contrastive learning (FILIP)
    decoupled_contrastive_learning = True,  # use decoupled contrastive learning (DCL) objective function, removing positive pairs from the denominator of the InfoNCE loss (CLOOB + DCL)
    extra_latent_projection = True,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
    use_visual_ssl = True,                  # whether to do self supervised learning on iages
    use_mlm = False,                        # use masked language learning (MLM) on text (DeCLIP)
    text_ssl_loss_weight = 0.05,            # weight for text MLM loss
    image_ssl_loss_weight = 0.05            # weight for image self-supervised learning loss
)
clip.to(device)
token = tokenizer.tokenize(sentence)
images = torch.zeros(3, 3, 256, 256)
images[0,:,5:10,5:10] = 255
images[1,:,:2:50,0:2] = 255
images[2,:,0:2,2:50] = 255

for time in trange(10000):
    loss = clip(
        token.cuda(),
        images.cuda(),
        freeze_image_encoder = False,   # whether to freeze image encoder if using a pretrained image net, proposed by LiT paper
        return_loss = True              # needs to be set to True to return contrastive loss
    )
    if (time%100 == 0):
        print(time,loss.detach().item())
    loss.backward()