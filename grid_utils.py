import numpy as np
from PIL import Image
import os
import glob
from copy import deepcopy
from tqdm import trange
WALL_VALUE = 200
MIN_ROOM_SIZE = 2
MAX_ROOM_SIZE = 4
MIN_REGION_SIZE = 4
MAX_REGION_SIZE = 5 * (MAX_ROOM_SIZE - 1) -1

def create_dir(name):
    try:
        os.mkdir(name)
    except:
        print(f'warning: {name} existed!')

def create_random_square_room(old_grid,old_mask,size):
    grid = deepcopy(old_grid)
    mask = deepcopy(old_mask)
    (grid_size,grid_size) = grid.shape
    #---------------------------------------------------------#        
    posX,posY = None,None
    for _ in range(100):
        tmp_posX = np.random.randint(size,grid_size-size)
        tmp_posY = np.random.randint(size,grid_size-size)
        failed = False
        for x in range(grid_size):
            if (failed):
                break
            for y in range(grid_size):
                if (abs(x-tmp_posX)<=size and abs(y-tmp_posY)<=size and mask[x,y]):
                    failed = True
                    break
        if (not failed):
            posX = tmp_posX
            posY = tmp_posY
            break
    #---------------------------------------------------------#        
    if (posY is None):
        return None,None
    #---------------------------------------------------------#        
    for x in range(grid_size):
        for y in range(grid_size):
            if (abs(x-posX)==size and abs(y-posY)<=size):
                grid[x,y] = WALL_VALUE
            if (abs(x-posX)<=size and abs(y-posY)==size):
                grid[x,y] = WALL_VALUE
            if (abs(x-posX)<=size+1 and abs(y-posY)<=size+1):
                mask[x,y] = 1
    return grid,mask

def create_random_diamond_room(old_grid,old_mask,size):
    grid = deepcopy(old_grid)
    mask = deepcopy(old_mask)
    (grid_size,grid_size) = grid.shape
    #---------------------------------------------------------#        
    posX,posY = None,None
    for _ in range(100):
        tmp_posX = np.random.randint(size,grid_size-size)
        tmp_posY = np.random.randint(size,grid_size-size)
        failed = False
        for x in range(grid_size):
            if (failed):
                break
            for y in range(grid_size):
                if (abs(x-tmp_posX)+abs(y-tmp_posY)<=size and mask[x,y]):
                    failed = True
                    break
        if (not failed):
            posX = tmp_posX
            posY = tmp_posY
            break
    #---------------------------------------------------------#        
    if (posY is None):
        return None,None
    #---------------------------------------------------------#        
    for x in range(grid_size):
        for y in range(grid_size):
            if (abs(x-posX)+abs(y-posY)==size):
                grid[x,y] = WALL_VALUE
            if (abs(x-posX)+abs(y-posY)<=size+1):
                mask[x,y] = 1
    return grid,mask

def create_empty_grid(grid):
    grid[:,0:1] = WALL_VALUE
    grid[:,-1:] = WALL_VALUE
    grid[0:1,:] = WALL_VALUE
    grid[-1:,:] = WALL_VALUE

    return grid
