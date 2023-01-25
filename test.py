import numpy as np
from PIL import Image
import os
import glob
from copy import deepcopy
from tqdm import trange
def create_dir(name):
    try:
        os.mkdir(name)
    except:
        print(f'warning: {name} existed!')
mode = '2_step_test'
num_sample = np.asarray([5000,10000])*1
# mode = 'valid'
WALL_VALUE = 200
root = f'./dataset'
create_dir(root)
create_dir(f'{root}/{mode}')

def create_random_room(old_grid):
    grid = deepcopy(old_grid)
    (grid_size,grid_size) = grid.shape
    posX,posY = None,None

    list_position_prompts = [
        'Add a room in the top left',
        'Add a room in the top right',
        'Add a room in the bottom left',
        'Add a room in the bottom right',
    ]
    position_promt = np.random.choice(range(len(list_position_prompts)))
    if position_promt == 0:
        posX = np.random.randint(low=1,high=grid_size//2-1)
        posY = np.random.randint(low=1,high=grid_size//2-1)
    elif position_promt == 1:
        posX = np.random.randint(low=1,high=grid_size//2-1)
        posY = np.random.randint(low=grid_size//2+1,high=grid_size-1)
    elif position_promt == 2:
        posX = np.random.randint(low=grid_size//2+1,high=grid_size-1)
        posY = np.random.randint(low=1,high=grid_size//2-1)
    elif position_promt == 3:
        posX = np.random.randint(low=grid_size//2-1,high=grid_size-1)
        posY = np.random.randint(low=grid_size//2-1,high=grid_size-1)

    #---------------------------------------------------------#        
    size = np.random.randint(low=2,high=5)
    #---------------------------------------------------------#        
    for x in range(grid_size):
        for y in range(grid_size):
            if (abs(x-posX)+abs(y-posY)==size):
                grid[x,y] = WALL_VALUE

    final_promt = list_position_prompts[position_promt] + f' with size {size}'
    return grid,final_promt

def create_empty_grid(grid):
    grid[:,0:1] = WALL_VALUE
    grid[:,-1:] = WALL_VALUE
    grid[0:1,:] = WALL_VALUE
    grid[-1:,:] = WALL_VALUE

    return grid

def save_grid(grid,id):
    name = f'{id}'
    while len(name)<7:
        name = '0'+name
    im = Image.fromarray(grid)
    im.save(f"{root}/{mode}/{name}.png")
    id += 1
    return id

def main(grid_size):
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
            new_grid, promt = create_random_room(grid)
            name = f'{current_id}'
            while len(name)<7:
                name = '0'+name
            csv_file.write(f'{random_file.split("/")[-1]},{promt},{name}.png\n')
            current_id = save_grid(new_grid,current_id)
    csv_file.close()

if __name__ == '__main__':
    for size in [16]:
        main(grid_size=size)