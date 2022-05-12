import math
from pickle import TRUE
import cv2
import numpy as np
import random
import time
from tqdm import tqdm
import csv
import matplotlib
import matplotlib.pyplot as plt
import sys

matplotlib.use('TkAgg')
matplotlib.rcParams['figure.figsize'] = (18, 5)

plotting = True if len(sys.argv) > 1 and sys.argv[1] == 'plot' else False

motion_model = [[ 1,  0, 1],
                [ 0,  1, 1],
                [-1,  0, 1],
                [ 0, -1, 1],
                [-1, -1, 2**0.5],
                [-1,  1, 2**0.5],
                [ 1, -1, 2**0.5],
                [ 1,  1, 2**0.5]]

class Node:
    def __init__(self, x,y, cost, parent_idx):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent_idx = parent_idx #index of previous node
    
def Dijkstra(st_x, st_y, en_x, en_y, xlims, ylims, x_width, omap):

    start_node = Node(
        round(st_x - xlims[0]), 
        round(st_y - ylims[0]), 
        cost = 0, 
        parent_idx = -1
    )

    end_node = Node(
        round(en_x - xlims[0]), 
        round(en_y - ylims[0]), 
        cost = 0, 
        parent_idx = -1
    )

    open_set, closed_set = dict(), dict()
    open_set[(start_node.y - ylims[0]) * x_width + (start_node.x - xlims[0])] = start_node

    while True:
        #Break if there are no valid nodes to traverse in open set
        if not bool(open_set):
            no_path = True
            break
        
        curr_idx = min(open_set, key=lambda o: open_set[o].cost)
        current = open_set[curr_idx]

        if current.x == end_node.x and current.y == end_node.y:
            no_path = False
            end_node.parent_idx = current.parent_idx
            end_node.cost = current.cost
            break

        del open_set[curr_idx]
        closed_set[curr_idx] = current

        for move_x, move_y, move_cost in motion_model:
            node = Node(
                current.x + move_x,
                current.y + move_y,
                cost = current.cost + move_cost,
                parent_idx = curr_idx
            )

            new_idx = (node.y - ylims[0]) * x_width + (node.x - xlims[0])

            if new_idx in closed_set:
                continue
                
            if not node_in_obstacle(node, omap, xlims, ylims):
                continue

            if new_idx not in open_set:
                open_set[new_idx] = node
            else: 
                if open_set[new_idx].cost >= node.cost:
                    open_set[new_idx] = node

    rx, ry = [end_node.x + xlims[0]], [end_node.y + ylims[0]]
    r_parent_idx = end_node.parent_idx

    while r_parent_idx != -1:
        n = closed_set[r_parent_idx]
        rx.append(n.x + xlims[0])
        ry.append(n.y + ylims[0])
        r_parent_idx = n.parent_idx

    indices = np.array([rx, ry])
    
    return indices, no_path


def node_in_obstacle(node, omap, xlims, ylims):
    px = node.x + xlims[0]
    py = node.y + ylims[0]

    if px < xlims[0]:
        return False
    if py < ylims[0]:
        return False
    if px >= xlims[1]:
        return False
    if py >= ylims[1]:
        return False

    if omap[node.x,node.y]:
        return False

    return True

def generate_map(map_size, obst_num):
    space = np.zeros((map_size,map_size))
    ob_r_size = int(sys.argv[4]) if len(sys.argv) > 4 else 10

    space[:,0] = 1
    space[0,:] = 1
    space[:,map_size-1] = 1
    space[map_size-1,:] = 1

    for i in range(obst_num):
        randx = random.randint(0,map_size-1)
        randy = random.randint(0,map_size-1)
        space[randx-random.randint(0,ob_r_size):randx+random.randint(0,ob_r_size), randy-random.randint(0,ob_r_size):randy+random.randint(0,ob_r_size)] = 1

    free_space = np.array(np.where(space == 0)).T
    
    point_a = free_space[random.randint(0, len(free_space)//2)]
    point_b = free_space[random.randint(len(free_space)//2, len(free_space)-1)]

    if random.random() > 0.5:
        start_point = point_a
        end_point = point_b
    else:
        end_point = point_a
        start_point = point_b

    if len(start_point) == 0:
        print("Null case?")

    if plotting:
        plotsp = space.copy()
        plotsp[start_point[0], start_point[1]] = 2
        plotsp[end_point[0], end_point[1]] = 3
        plt.imshow(plotsp)
        plt.show()

    return space, start_point, end_point

def plan_optimal_path(map_size, obst_num, no_path_cases):
    omap, start_point, end_point = generate_map(map_size, obst_num)

    indices, no_path = Dijkstra(
        start_point[0], 
        start_point[1],
        end_point[0],
        end_point[1],
        [0, len(omap)],
        [0, len(omap)],
        len(omap),
        omap
    )
    
    no_path_cases += no_path

    if plotting:
        omap[indices[0], indices[1]] = 2
        plt.imshow(omap)
        plt.show()        
    
    omap[start_point[0], start_point[1]] = 2
    omap[end_point[0], end_point[1]] = 3
    
    path = np.zeros_like(omap)
    path[indices[0], indices[1]] = 1

    omap = (omap.ravel()).astype(int)
    omap = omap.tolist()
    
    path = (path.ravel()).astype(int)
    path = path.tolist()

    return omap, path, no_path_cases

def main():
    num_data = 50000
    no_path_cases = 0
    map_size = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    obst_num = int(sys.argv[3]) if len(sys.argv) > 3 else 20

    st = time.time()

    iwtr = csv.writer(open('data/inputs.csv', 'w'), delimiter=',', lineterminator='\n')
    owtr = csv.writer(open('data/outputs.csv', 'w'), delimiter=',', lineterminator='\n')
    
    for i in tqdm(range(num_data)):
        omap, path, no_path_cases = plan_optimal_path(map_size, obst_num, no_path_cases)

        iwtr.writerows([omap])
        owtr.writerows([path])

    en = time.time()

    print("Done.")
    print("Time to generate {} maps: ".format(num_data) + str(np.round(en - st, 3)) + "seconds. \n No path cases: {}".format(no_path_cases))

if __name__ == '__main__':
    main()

