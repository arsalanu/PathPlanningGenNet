import sys
from tqdm import tqdm
from time import time
import torch
import torch.nn as nn
import torch.utils as utils
import numpy as np
import matplotlib
import matplotlib.pyplot as pl

matplotlib.use('TkAgg')
matplotlib.rcParams['figure.figsize'] = (22, 10)
# torch.manual_seed(1234)
# np.random.seed(1234)

from preproc import preprocess
from pathgen import PathGen
from pgen_loss import modified_mse, modified_mae, obst_mod_mae, obst_mod_mse
from training import training

def main():

    #Input data map size
    map_size = int(sys.argv[1]) if len(sys.argv) > 1 else 64

    #Train-val data split
    split = 0.98

    #Network training parameters
    batch_size = 16
    epochs = 15
    learning_rate = 1e-3

    print("Preprocessing...")

    #Create iterators for training and validation dataset
    tr_dataloader, val_dataloader = preprocess(
        input_file='PATHGEN_NET/data/inputs.csv',
        output_file='PATHGEN_NET/data/outputs.csv',
        map_size=map_size,
        split=split,
        batch_size=batch_size
    )

    #Initialise network architecture
    model = PathGen()

    print("Training...")

    #Train (and validate) the model
    model = training(
        model,
        tr_dataloader,
        val_dataloader,
        epochs,
        learning_rate,
        loss_fn=modified_mse,
        batch_size=batch_size,
        map_size=map_size,
        plot=False
    )

    #Save trained model
    saved_model_path = 'PATHGEN_NET/PathGen_64_Large.pth'
    torch.save(model.state_dict(), saved_model_path)

if __name__ == '__main__':
    main()