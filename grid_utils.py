import numpy as np
from PIL import Image
import os
import glob
from copy import deepcopy
from tqdm import trange
WALL_VALUE = 200
MIN_ROOM_SIZE = 2
MAX_ROOM_SIZE = 4

def create_dir(name):
    try:
        os.mkdir(name)
    except:
        print(f'warning: {name} existed!')

def create_random_square_room(old_grid,size,Xmin,Ymin,Xmax,Ymax):
    grid = deepcopy(old_grid)
    (grid_size,grid_size) = grid.shape
    #---------------------------------------------------------#        
    posX,posY = None,None
    for _ in range(100):
        tmp_posX = np.random.randint(Xmin+size,Xmax-size)
        tmp_posY = np.random.randint(Ymin+size,Ymax-size)
        failed = False
        for x in range(grid_size):
            if (failed):
                break
            for y in range(grid_size):
                if (abs(x-tmp_posX)==size and abs(y-tmp_posY)<=size):
                    if (grid[x,y] == WALL_VALUE):
                        failed = True
                        break
                elif (abs(x-tmp_posX)<=size and abs(y-tmp_posY)==size):
                    if (grid[x,y] == WALL_VALUE):
                        failed = True
                        break
        if (not failed):
            posX = tmp_posX
            posY = tmp_posY
            break
    #---------------------------------------------------------#        
    if (posY is None):
        return None
    #---------------------------------------------------------#        
    for x in range(grid_size):
        for y in range(grid_size):
            if (abs(x-tmp_posX)==size and abs(y-tmp_posY)<=size):
                grid[x,y] = WALL_VALUE
            elif (abs(x-tmp_posX)<=size and abs(y-tmp_posY)==size):
                grid[x,y] = WALL_VALUE
    return grid

def create_random_diamond_room(old_grid,size,Xmin,Ymin,Xmax,Ymax):
    grid = deepcopy(old_grid)
    (grid_size,grid_size) = grid.shape
    #---------------------------------------------------------#        
    posX,posY = None,None
    for _ in range(100):
        tmp_posX = np.random.randint(Xmin+size,Xmax-size)
        tmp_posY = np.random.randint(Ymin+size,Ymax-size)
        failed = False
        for x in range(grid_size):
            if (failed):
                break
            for y in range(grid_size):
                if (abs(x-tmp_posX)+abs(y-tmp_posY)<=size and grid[x,y]==WALL_VALUE):
                    failed = True
                    break
        if (not failed):
            posX = tmp_posX
            posY = tmp_posY
            break
    #---------------------------------------------------------#        
    if (posY is None):
        return None
    #---------------------------------------------------------#        
    for x in range(grid_size):
        for y in range(grid_size):
            if (abs(x-posX)+abs(y-posY)==size):
                grid[x,y] = WALL_VALUE
    return grid

def create_empty_grid(grid):
    grid[:,0:1] = WALL_VALUE
    grid[:,-1:] = WALL_VALUE
    grid[0:1,:] = WALL_VALUE
    grid[-1:,:] = WALL_VALUE

    return grid
