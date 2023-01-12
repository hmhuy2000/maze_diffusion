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

def create_border(grid):
    grid[:,0] = 156
    grid[:,-1] = 156
    grid[0,:] = 156
    grid[-1,:] = 156

    return grid

def main(grid_size,target_image_size):
    grid = np.zeros((grid_size,grid_size),dtype=np.uint8)
    grid = create_border(grid)

    im = Image.fromarray(padding(grid,target_image_size))
    im.save(f"test_figs/grid_{grid_size}.jpg")

if __name__ == '__main__':
    for size in [16,32,64,128,256,512]:
        main(grid_size=size,target_image_size=size)