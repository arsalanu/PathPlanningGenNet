import torch
import torch.nn as nn
import numpy as np

def modified_mse(omap, pred, y, batch_size=8):
    free_error = y[y == 0] - pred[y == 0]
    obst_error = y[y == 1] - pred[y != 0]

    z_sum = (torch.sum(torch.pow(free_error, 2)) / len(free_error)) / batch_size 
    a_sum = (torch.sum(torch.pow(obst_error, 2)) / len(obst_error)) / batch_size

    return z_sum * 0.5 + a_sum * 0.5

def modified_mae(omap, pred, y, batch_size=8):
    free_error = y[y == 0] - pred[y == 0]
    obst_error = y[y == 1] - pred[y != 0]

    z_sum = (torch.sum(torch.abs(free_error)) / len(free_error)) / batch_size
    a_sum = (torch.sum(torch.abs(obst_error)) / len(obst_error)) / batch_size

    return z_sum * 0.5 + a_sum * 0.5

def obst_mod_mse(omap, pred, y, batch_size=8):
    pred = pred * (1- torch.unsqueeze(omap[:,0],1))

    free_error = y[y == 0] - pred[y == 0]
    obst_error = y[y == 1] - pred[y != 0]

    z_sum = (torch.sum(torch.pow(free_error, 2)) / len(free_error)) / batch_size 
    a_sum = (torch.sum(torch.pow(obst_error, 2)) / len(obst_error)) / batch_size

    return z_sum * 0.5 + a_sum * 0.5

def obst_mod_mae(omap, pred, y, batch_size=8):
    pred = pred * (1 - torch.unsqueeze(omap[:,0],1))

    free_error = y[y == 0] - pred[y == 0]
    obst_error = y[y == 1] - pred[y != 0]

    z_sum = (torch.sum(torch.pow(free_error, 2)) / len(free_error)) / batch_size 
    a_sum = (torch.sum(torch.pow(obst_error, 2)) / len(obst_error)) / batch_size

    return z_sum * 0.5 + a_sum * 0.5 
