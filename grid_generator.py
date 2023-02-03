import numpy as np
from PIL import Image
import os
import glob
from copy import deepcopy
from tqdm import trange

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
# mode = 'train'
# mode = 'test'
if (mode == 'train'):
    num_scale = 10
if (mode == 'train_small'):
    num_scale = 3
mode = f'64_3step_region_{mode}'
num_sample = np.asarray([5000,10000,10000])*num_scale
print(f'create dataset as {mode} with num_sample = {num_sample}')
root = f'./dataset'

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
        size = 8
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
        num_room = np.random.randint(1,3+1)
        valid_room_size = size

        use_square = 0 #np.random.randint(0,100)%2==0
        create_room_fn = None
        if (use_square):
            create_room_fn = create_random_square_room
            prompt = f'create {num_room} square,{centerX-size},{centerY-size},{centerX+size},{centerY+size}'
        else:
            create_room_fn = create_random_diamond_room
            prompt = f'create {num_room} diamond,{centerX-size},{centerY-size},{centerX+size},{centerY+size}'

        for id in range(num_room):
            for trial_room_place in range(100):
                room_size = np.random.randint(MIN_ROOM_SIZE,min(MAX_ROOM_SIZE,valid_room_size - (num_room-id-1)*MIN_ROOM_SIZE)+1)
                tmp_grid = None
                tmp_grid = create_room_fn(grid,room_size,
                                centerX-size,centerY-size,
                                centerX+size,centerY+size)

                if (tmp_grid is not None):
                    valid_room_size -= room_size
                    grid = tmp_grid
                    break

        return grid,prompt

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