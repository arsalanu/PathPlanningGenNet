import numpy as np
from sklearn import cluster
import torch
import torch.nn as nn
from skimage import measure
import matplotlib
import matplotlib.pyplot as plt

# torch.manual_seed(1234)
# np.random.seed(1234)

matplotlib.rcParams['figure.figsize'] = (22, 10)

def plot_maps(pp, xx, yy, st, en, count):

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    pp[st[0],st[1]] = 1
    pp[en[0],en[1]] = 1

    plt.subplot(2,4,1)
    plt.title('Obstacle map', size=10)
    plt.imshow(xx[0], cmap='magma')
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(2,4,2)
    plt.title('Ground truth', size=10)
    plt.imshow(xx[0] + (yy+1) + xx[1] + xx[2], cmap='magma')
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(2,4,3)
    plt.title('Raw output path', size=10)
    plt.imshow(xx[0] + pp * 2 + xx[1] + xx[2], cmap='magma')
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(2,4,4)
    plt.title('Cond. output path', size=10)
    plt.imshow(xx[0] + ((pp > 0.5) * pp) * 2 + xx[1] + xx[2], cmap='magma')
    plt.colorbar(fraction=0.046, pad=0.04)

    existing = (pp > 0.45) * 1.
    existing = existing * (1 - xx[0])
    existing = (xx[0] + 1) + + xx[1] + xx[2] + existing * 2 

    plt.subplot(2,4,5)
    plt.title('Output path $C > 0.4$', size=10)
    plt.imshow(existing, cmap='magma')
    plt.colorbar(fraction=0.046, pad=0.04)

    existing = (pp > 0.6) * 1.
    existing = existing * (1 - xx[0])
    existing = (xx[0] + 1) + + xx[1] + xx[2]+ existing * 2

    plt.subplot(2,4,6)
    plt.title('Output path $C > 0.5$', size=10)
    plt.imshow(existing, cmap='magma')
    plt.colorbar(fraction=0.046, pad=0.04)

    existing = (pp > 0.75) * 1.
    existing = existing * (1 - xx[0])
    existing = (xx[0] + 1) + + xx[1] + xx[2] + existing * 2

    plt.subplot(2,4,7)
    plt.title('Output path $C > 0.6$', size=10)
    plt.imshow(existing, cmap='magma')
    plt.colorbar(fraction=0.046, pad=0.04)

    existing = (pp > 0.85) * 1.
    existing = existing * (1 - xx[0])
    existing = (xx[0] + 1) + xx[1] + xx[2] + existing * 2

    plt.subplot(2,4,8)
    plt.title('Output path $C > 0.7$', size=10)
    plt.imshow(existing, cmap='magma')
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.savefig('PATHGEN_NET/examples/map_{}.png'.format(count))
    # plt.pause(0.2)

    plt.clf()

def postproc(pred_p, gt_p, in_map, st_pt, en_pt, score_thres, plot, count):
    
    if plot: plot_maps(pred_p, in_map, gt_p, st_pt, en_pt, count)

    #Remove obstacle overlap from the predicted path
    pred_p = pred_p * (1 - in_map[0])

    #Apply threshold to the predicted path
    pred_p[pred_p < score_thres] = 0

    #Get tuple of indices of all non-zero (possible path) regions
    connected = measure.label(pred_p != 0)
    num_clusters = np.max(connected)

    #Create dictionary of sets of non-zero clusters in image
    clusters = {}

    for clus in range(1, num_clusters+1):
        clus_list = np.argwhere(connected == clus)
        clusters[clus] = set([tuple(list(i)) for i in clus_list])

    #Store the indices of the path
    gt_indices = set([tuple(list(i)) for i in np.argwhere(gt_p)])

    #Check for all clusters if start and end point are in the cluster
    for cset in clusters:
        valid_path = False
        optimal_in_region = None
        search_space_reduction = None

        #Check if start point and end point are in the cluster
        if tuple(list(st_pt)) in clusters[cset] and tuple(list(en_pt)) in clusters[cset]:
            valid_path = True

        #Percentage of the total space we need to search to find optimal region
        if valid_path:

            #Check if whole path is in the cluster
            if gt_indices.issubset(clusters[cset]):   
                optimal_in_region = 100
            else:
                opt_ct = 0
                for i in gt_indices:
                    if i in clusters[cset]:
                        opt_ct += 1
            
            optimal_in_region = (opt_ct/len(gt_indices)) * 100

            #Evaluate search space reduction
            search_space_reduction = (len(clusters[cset])/np.sum(in_map != 1)) * 100
            break

    return valid_path, optimal_in_region, search_space_reduction




