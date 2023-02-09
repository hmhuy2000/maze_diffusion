import numpy as np
from PIL import Image
import os
import glob
from copy import deepcopy
from tqdm import trange
import random

import psutil
print(f'Number of CPUs: {psutil.cpu_count()}')
p = psutil.Process()

arr_cpus = [i for i in range(80,96)]

p.cpu_affinity(arr_cpus)
print(f'CPU pool after assignment ({len(arr_cpus)}): {p.cpu_affinity()}')
import warnings
warnings.filterwarnings("ignore")

from grid_utils import *
num_scale = 1
mode = 'train_small'
mode = 'train'
# mode = 'test'
if (mode == 'train'):
    num_scale = 10
if (mode == 'train_small'):
    num_scale = 5
mode = f'square_region_{mode}'
num_sample = np.asarray([5000,10000,10000])*num_scale
print(f'create dataset as {mode} with num_sample = {num_sample}')
root = f'./dataset'

def create_objects(grid,mask,square,diamond):
    if (grid is None):
        return None,None
    if (square+diamond==0):
        return grid,mask
    tmp = []
    use_square,use_diamond = 0,0
    for _ in range(square):
        tmp.append('square')
    for _ in range(diamond):
        tmp.append('diamond')
    chosen = np.random.choice(tmp)
    (grid_size,grid_size) = grid.shape
    create_fn = None
    if (chosen == 'diamond'):
        create_fn = create_random_diamond_room
        use_diamond = 1
    if (chosen == 'square'):
        create_fn = create_random_square_room
        use_square = 1

    for size in range(MIN_ROOM_SIZE,MAX_ROOM_SIZE+1)[::-1]:
        tmp_grid,tmp_mask = create_fn(grid,mask,size)
        return_grid,return_mask = create_objects(tmp_grid,tmp_mask,square-use_square,diamond-use_diamond)
        if (return_grid is not None):    
            return return_grid,return_mask

    return None,None

def save_grid(grid,id):
    name = f'{id}'
    while len(name)<7:
        name = '0'+name
    im = Image.fromarray(grid)
    im.save(f"{root}/{mode}/{name}.png")
    id += 1
    return id

def random_region_generated_dataset(input_grid):
    grid = deepcopy(input_grid)
    (grid_size,grid_size) = grid.shape
    for trial_pos in range(100):
        size = 10
        centerX = np.random.randint(size,grid_size-size)
        centerY = np.random.randint(size,grid_size-size)
        prompt = None
        failed = False
        for x in range(grid_size):
            if (failed):
                break
            for y in range(grid_size):
                if (abs(x-centerX)<=size and abs(y-centerY)<=size and grid[x,y]==WALL_VALUE):
                    failed=True
                    break

        if (failed):
            continue
        #-----------------------------------------------#

        num_room = np.random.randint(1,3+1)
        use_square = num_room #np.random.randint(0,num_room+1)
        use_diamond = num_room - use_square
        obj_ls = []
        if (use_square):
            obj_ls.append((use_square,'square'))
        if (use_diamond):
            obj_ls.append((use_diamond,'diamond'))
        prompt = 'Create'
        for id,(num,key) in enumerate(obj_ls):
            if (num>1):
                key = key + 's'
            if (id == 0):
                prompt = prompt + f' {num_dict[num]} {key}'
            else:
                prompt = prompt + f' and {num_dict[num]} {key}'
        prompt = prompt + f',{centerX-size},{centerY-size},{centerX+size},{centerY+size}'
        #-----------------------------------------------#

        tmp_grid,tmp_mask = create_objects(grid=grid[centerX-size:centerX+size+1,centerY-size:centerY+size+1],
                                  mask=np.zeros(grid[centerX-size:centerX+size+1,centerY-size:centerY+size+1].shape),
                                  square=use_square,diamond=use_diamond)
        
        if (tmp_grid is not None):
            grid[centerX-size:centerX+size+1,centerY-size:centerY+size+1] = tmp_grid
            return grid,prompt
    return None, None

def main(grid_size):
    create_dir(root)
    create_dir(f'{root}/{mode}')

    csv_file = open(f'{root}/dataset_{mode}.csv','w')
    current_id = 0
    grid = np.zeros((grid_size,grid_size),dtype=np.uint8)
    grid = create_empty_grid(grid)
    current_id = save_grid(grid,current_id)
    for loop in range(len(num_sample)):
        list_file = glob.glob(f"{root}/{mode}/*.png")
        for time in trange(num_sample[loop]):
            new_grid, prompt = None,None
            random_file = None
            while (new_grid is None):
                random_file = list_file[np.random.randint(0,len(list_file))]
                grid = np.asarray(Image.open(random_file))
                new_grid, prompt = random_region_generated_dataset(grid)
            name = f'{current_id}'
            while len(name)<7:
                name = '0'+name
            csv_file.write(f'{random_file.split("/")[-1]},{prompt},{name}.png\n')
            current_id = save_grid(new_grid,current_id)
    csv_file.close()

if __name__ == '__main__':
    for size in [64]:
        main(grid_size=size)