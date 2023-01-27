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
exp_name = '3_step_test'
num_sample = np.asarray([100,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200,200])*1
WALL_VALUE = [200,200,200]
AGENT_VALUE = [0,0,200]
GOAL_VALUE = [200,0,0]
MIN_ROOM_SIZE = 2
root = f'./dataset'
create_dir(root)
create_dir(f'{root}/{exp_name}')

def check_valid_grid(grid):
    (grid_size,grid_size,_) = grid.shape
    agentX,agentY=None,None
    goalX,goalY=None,None
    for x in range(grid_size):
        for y in range(grid_size):
            if (grid[x,y,0] == AGENT_VALUE[0] and grid[x,y,1] == AGENT_VALUE[1] and grid[x,y,2] == AGENT_VALUE[2]):
                agentX = x
                agentY = y
            if (grid[x,y,0] == GOAL_VALUE[0] and grid[x,y,1] == GOAL_VALUE[1] and grid[x,y,2] == GOAL_VALUE[2]):
                goalX = x
                goalY = y
    
    if (goalX is None or agentX is None):
        return False
    
    mask = np.zeros((grid_size,grid_size),dtype=np.uint8)
    mask[agentX,agentY] = 255
    queue = [(agentX,agentY)]
    dirrectX = [-1,1,0,0]
    dirrectY = [0,0,1,-1]
    while(len(queue)):
        (curX,curY) = queue.pop(0)
        for idx in range(len(dirrectX)):
            newX = curX + dirrectX[idx]
            newY = curY + dirrectY[idx]
            if (0<newX<grid_size and 0<newY<grid_size and mask[newX,newY]==0 and
             (grid[newX,newY,0]!=WALL_VALUE[0] or grid[newX,newY,1]!=WALL_VALUE[1] or grid[newX,newY,2]!=WALL_VALUE[2])):
                if (newX == goalX and newY == goalY):
                    return True
                mask[newX,newY] = 255
                queue.append((newX,newY))
    return False
    
def create_random_square_room(old_grid,size,Xmin,Ymin,Xmax,Ymax):
    cnt = 0
    while(True):
        cnt += 1
        if (cnt == 10):
            return None
        # tmp_old = deepcopy(old_grid)
        # img = Image.fromarray(tmp_old)
        # img.save('old.png')
        grid = deepcopy(old_grid)
        (grid_size,grid_size,_) = grid.shape
        posX = np.random.randint(Xmin,Xmax+1)
        posY = np.random.randint(Ymin,Ymax+1)

        #---------------------------------------------------------#        
        failed = False
        doors = []
        for _ in range(np.random.randint(1,size//2+1)):
            doorX,doorY=None,None
            if (np.random.randint(1,11)<=5):
                if (np.random.randint(1,11)<=5):
                    doorX = posX - size
                else:
                    doorX = posX + size
                doorY = np.random.randint(posY - size + 1,posY + size - 1)
            else:
                if (np.random.randint(1,11)<=5):
                    doorY = posY - size
                else:
                    doorY = posY + size
                doorX = np.random.randint(posX - size + 1,posX + size - 1)
            doors.append((doorX,doorY))
        
        for x in range(grid_size):
            if (failed):
                break
            for y in range(grid_size):
                if (abs(x-posX)==size and abs(y-posY)<=size):
                    if (np.min(grid[x,y]) == np.min(WALL_VALUE)):
                        failed = True
                        break
                    grid[x,y] = WALL_VALUE
                elif (abs(x-posX)<=size and abs(y-posY)==size):
                    if (np.min(grid[x,y]) == np.min(WALL_VALUE)):
                        failed = True
                        break
                    grid[x,y] = WALL_VALUE
                if ((x,y) in doors):
                    grid[x,y] = 0
        # img = Image.fromarray(grid)
        # img.save('new.png')
        if (not failed and check_valid_grid(grid=grid)):
            return grid

def create_empty_grid(grid):
    grid[:,0:1] = WALL_VALUE
    grid[:,-1:] = WALL_VALUE
    grid[0:1,:] = WALL_VALUE
    grid[-1:,:] = WALL_VALUE

    return grid

def place_random_agent_goal(grid):
    (grid_size,grid_size,_) = grid.shape
    agentX = np.random.randint(1,grid_size-1)
    agentY = np.random.randint(1,grid_size-1)
    
    while(True):
        goalX = np.random.randint(1,grid_size-1)
        goalY = np.random.randint(1,grid_size-1)
        if ((goalX,goalY) != (agentX,agentY)):
            break
    
    grid[agentX,agentY]=AGENT_VALUE
    grid[goalX,goalY]=GOAL_VALUE

    return grid

def save_grid(grid,id):
    name = f'{id}'
    while len(name)<7:
        name = '0'+name
    im = Image.fromarray(grid)
    im.save(f"{root}/{exp_name}/{name}.png")
    id += 1
    return id

def random_generate_dataset(input_grid):
    grid = deepcopy(input_grid)
    (grid_size,grid_size,_) = grid.shape
    padding = 5
    for tmp_cnt in range(10):
        random_centerX = np.random.randint(padding,grid_size-padding)
        random_centerY = np.random.randint(padding,grid_size-padding)
        best_size=0
        largest_size = min(min(grid_size-random_centerX,random_centerX),min(grid_size-random_centerY,random_centerY))
        for size in range(largest_size-1,MIN_ROOM_SIZE,-1):
            best_size=size
            for x in range(grid_size):
                if (best_size==-1):
                    break
                for y in range(grid_size):
                    if (abs(x-random_centerX)==size and abs(y-random_centerY)<=size and np.min(grid[x,y])==np.min(WALL_VALUE)):
                        best_size=-1
                        break
                    if (abs(x-random_centerX)<=size and abs(y-random_centerY)==size and np.min(grid[x,y])==np.min(WALL_VALUE)):
                        best_size=-1
                        break
            if (best_size!=-1):
                break
        if (best_size==-1):
            continue
        size = np.random.randint(MIN_ROOM_SIZE,best_size+1)
        num_room = 1 + np.random.randint(0,size//5+1)
        promt = f'Add {num_room} rooms into region from ({random_centerX-size} {random_centerY-size}) to ({random_centerX+size} {random_centerY+size})'
        valid_room_size = size
        for id in range(num_room):
            if (best_size == -1):
                break
            room_size=None
            for trial in range(10):
                room_size = valid_room_size - (num_room-id-1)*MIN_ROOM_SIZE
                tmp_grid = create_random_square_room(grid,room_size,random_centerX-size,random_centerY-size,
                        random_centerX+size,random_centerY+size)
                if (tmp_grid is not None):
                    grid = tmp_grid
                    break
                if (trial>10):
                    best_size = -1
                    break
            valid_room_size -= room_size

        if (best_size==-1):
            continue
        return grid,promt
    return None,None

def main(grid_size):
    csv_file = open(f'{root}/dataset_{exp_name}.csv','w')
    current_id = 0
    empty_grid = np.zeros((grid_size,grid_size,3),dtype=np.uint8)
    empty_grid = create_empty_grid(empty_grid)
    current_id = save_grid(empty_grid,current_id)

    #----------------------------------------------------------------------#

    list_file = glob.glob(f"{root}/{exp_name}/*.png")
    random_file = list_file[np.random.randint(0,len(list_file))]
    for _ in trange(num_sample[0]):
        grid = deepcopy(empty_grid)
        grid = place_random_agent_goal(grid)
        current_id = save_grid(grid,current_id)
        name = f'{current_id}'
        while len(name)<7:
            name = '0'+name
        promt = 'Place random agent'
        csv_file.write(f'{random_file.split("/")[-1]},{promt},{name}.png\n')


    #----------------------------------------------------------------------#

    for loop in range(1,len(num_sample)):
        list_file = glob.glob(f"{root}/{exp_name}/*.png")
        list_file.pop(0)
        for time in trange(num_sample[loop]):
            random_file = list_file[np.random.randint(0,len(list_file))]
            grid = np.asarray(Image.open(random_file))
            new_grid, promt = random_generate_dataset(grid)
            if (new_grid is None):
                continue

            name = f'{current_id}'
            while len(name)<7:
                name = '0'+name
            csv_file.write(f'{random_file.split("/")[-1]},{promt},{name}.png\n')
            current_id = save_grid(new_grid,current_id)
    csv_file.close()

if __name__ == '__main__':
    for size in [64]:
        main(grid_size=size)