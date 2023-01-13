import numpy as np
from PIL import Image

def padding(img,img_size):
    toggle = 0
    while img.shape[0]<img_size:
        if (toggle == 0):
            img = np.concatenate((img,np.zeros((1,img.shape[1]),dtype=np.uint8)),0)
        else:
            img = np.concatenate((np.zeros((1,img.shape[1]),dtype=np.uint8),img),0)
        toggle = 1 - toggle
    toggle = 0
    while img.shape[1]<img_size:
        if (toggle == 0):
            img = np.concatenate((img,np.zeros((img.shape[0],1),dtype=np.uint8)),1)
        else:
            img = np.concatenate((np.zeros((img.shape[0],1),dtype=np.uint8),img),1)
        toggle = 1 - toggle
    return img

def create_empty_grid(grid_size,scale=3):
    grid = np.zeros((grid_size*scale,grid_size*scale),dtype=np.uint8)
    grid[:,0:scale] = 200
    grid[:,-scale:] = 200
    grid[0:scale,:] = 200
    grid[-scale:,:] = 200

    return grid

def main(grid_size):
    grid = create_empty_grid(grid_size)
    im = Image.fromarray(grid)
    im.save(f"test_figs/empty_{grid_size}.png")

if __name__ == '__main__':
    for size in [16]:
        main(grid_size=size)