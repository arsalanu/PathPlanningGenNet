from tqdm import tqdm
from time import time
import torch
import torch.nn as nn
import torch.utils as utils
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from postproc import postproc

matplotlib.use('TkAgg')
matplotlib.rcParams['figure.figsize'] = (22, 10)

def validate(model, val_dataloader, loss_fn, batch_size, map_size, 
                val_loss_per_epoch, val_acc_per_epoch, device, epoch, thres, plot):

    val_batch_loss = []
    val_batch_acc = []

    valid = 0
    optimality = 0
    space_reduction = 0
    failures = 0
    COUNT = 0
    
    for x,y in val_dataloader:
        x,y = x.to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(x, pred, y, batch_size= batch_size)
        val_batch_loss.append(loss.item())

        sum_acc = 0
        for p in range(len(pred)):
            acc = torch.sum(torch.round(pred[p]) == y[p]) / (map_size**2)
            sum_acc += acc.item()

            pp = pred[p].squeeze(0).cpu().detach().numpy()
            yy = y[p].squeeze(0).cpu().detach().numpy()
            xx = x[p].cpu().detach().numpy()

            st = np.argwhere(xx[1] == 1).squeeze(0)
            en = np.argwhere(xx[2] == 1).squeeze(0)

            if plot and epoch == 14:
                v, o, s = postproc(pp, yy, xx, st, en, score_thres=thres, plot=True, count=COUNT)
            else:
                v, o, s = postproc(pp, yy, xx, st, en, score_thres=thres, plot=False, count=COUNT)
                
            valid += v

            if o != None: optimality += o 
            if s != None: space_reduction += s

            if o == None and s == None: 
                failures += 1

            COUNT += 1

        b_acc=  sum_acc / len(pred)
        val_batch_acc.append(b_acc)
    
    val_loss_per_epoch.append(sum(val_batch_loss)/len(val_dataloader))
    val_acc_per_epoch.append(sum(val_batch_acc)/len(val_dataloader))
    
    val_valid = (valid / COUNT) * 100
    if COUNT - failures == 0:
        val_optimality = 0
        val_space_reduction = 0
    else:
        val_optimality = optimality / (COUNT - failures)
        val_space_reduction = space_reduction / (COUNT - failures)
    
    return val_loss_per_epoch, val_acc_per_epoch, val_valid, val_optimality, val_space_reduction

def training(model, tr_dataloader, val_dataloader, epochs, 
                learning_rate, loss_fn, batch_size, map_size, plot=False):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    loss_fn = loss_fn

    loss_per_epoch = []
    accuracy_per_epoch = []
    val_loss_per_epoch = []
    val_acc_per_epoch = []
    val_validity_per_epoch = {0.4:[], 0.5:[], 0.6:[]}
    val_optimality_per_epoch = {0.4:[], 0.5:[], 0.6:[]}
    val_ss_per_epoch = {0.4:[], 0.5:[], 0.6:[]}
    

    for epoch in range(epochs):
        model.train()
        batch_loss = []
        batch_accuracy = []

        for x,y in tqdm(tr_dataloader):
            x,y = x.to(device), y.to(device)

            pred = model(x)
            loss = loss_fn(x, pred, y, batch_size=batch_size)
            batch_loss.append(loss.item())

            sum_acc = 0
            for p in range(len(pred)):
                acc = torch.sum(torch.round(pred[p]) == y[p]) / (map_size**2)
                sum_acc += acc.item()

            b_acc=  sum_acc / len(pred)

            batch_accuracy.append(b_acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        loss_per_epoch.append(sum(batch_loss)/len(tr_dataloader))
        accuracy_per_epoch.append(sum(batch_accuracy)/len(tr_dataloader))

        # Show metrics
        print('Epoch {} \nTrain loss: {} | Train accuracy: {}'.format(
            epoch,
            loss_per_epoch[-1], 
            accuracy_per_epoch[-1]
        ))


        #*Validation
        model.eval()

        for thres in [0.4, 0.5, 0.6]:

            val_loss_per_epoch, val_acc_per_epoch, val_valid, val_optimality, val_space_reduction = validate(
                model, 
                val_dataloader,
                loss_fn,
                batch_size,
                map_size,
                val_loss_per_epoch,
                val_acc_per_epoch,
                device,
                epoch,
                thres,
                plot
            )

            # Show metrics
            print('THRES: {} | Val loss: {} | Val accuracy: {} \nVALIDITY: {}% | MEAN OPTIMALITY: {}% | MEAN SEARCH SPACE: {}% '.format(
                thres,
                val_loss_per_epoch[-1], 
                val_acc_per_epoch[-1],
                round(val_valid,2),
                round(val_optimality,2),
                round(val_space_reduction,2)
            ))

            val_validity_per_epoch[thres].append(val_valid)
            val_optimality_per_epoch[thres].append(val_optimality)
            val_ss_per_epoch[thres].append(val_space_reduction)


    # This can be used to write a log of metrics to an output file
    # with open("output_record.log", "w") as fp:
    #     fp.write("Training accuracy: " + "[" + ",".join([str(round(i,4)) for i in accuracy_per_epoch]) + "]" + "\n")
    #     fp.write("Training loss: " + "[" + ",".join([str(round(i,4)) for i in loss_per_epoch]) + "]" + "\n")
    #     fp.write("Validation accuracy: " + "[" + ",".join([str(round(i,4)) for i in val_acc_per_epoch]) + "]" + "\n")
    #     fp.write("Validation loss: " + "[" + ",".join([str(round(i,4)) for i in val_loss_per_epoch]) + "]" + "\n")
    #     fp.write(json.dumps(val_validity_per_epoch))
    #     fp.write(json.dumps(val_optimality_per_epoch))
    #     fp.write(json.dumps(val_ss_per_epoch))

    return model