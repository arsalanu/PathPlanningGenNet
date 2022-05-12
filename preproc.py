import numpy as np
import torch
import torch.utils as utils

def read_data(inp_file, out_file):
    inputs = np.loadtxt(inp_file, dtype=int, delimiter=',')
    outputs = np.loadtxt(out_file, dtype=int, delimiter=',')
    print("Data is read.")
    return inputs, outputs

def parse_data(in_row, out_row, map_size):
    in_map = np.array(np.split(in_row, map_size))

    start_loc = np.array(np.where(in_map == 2)).flatten()
    goal_loc = np.array(np.where(in_map == 3)).flatten()

    if len(start_loc) == 0 or len(goal_loc) == 0:
        return None, None, True

    start_dim = np.zeros_like(in_map)
    start_dim[start_loc[0], start_loc[1]] = 1

    end_dim = np.zeros_like(in_map)
    end_dim[goal_loc[0], goal_loc[1]] = 1

    in_map[start_loc] = 0
    in_map[goal_loc] = 0
    
    in_map = np.dstack((in_map, start_dim, end_dim)) * 1
    out_map = np.array(np.split(out_row, map_size)) * 1

    return in_map, out_map, False

def create_iterators(inputs, outputs, map_size, batch_size, split):
    in_maps = []
    out_maps = []
    
    for i in range(len(inputs)):
        in_map, out_map, skip = parse_data(inputs[i], outputs[i], map_size)

        if skip:
            continue

        in_maps.append(in_map)
        out_maps.append(out_map)

    in_maps = np.array(in_maps)
    out_maps = np.array(out_maps)

    in_maps_tensor = torch.Tensor(in_maps)
    out_maps_tensor = torch.Tensor(out_maps)
    
    in_maps_tensor = torch.swapaxes(in_maps_tensor, 3, 1)
    in_maps_tensor = torch.swapaxes(in_maps_tensor, 3, 2)
    out_maps_tensor = torch.unsqueeze(out_maps_tensor,1)

    tr_in_maps_tensor = in_maps_tensor[:int(len(in_maps_tensor) * split)]
    tr_out_maps_tensor = out_maps_tensor[:int(len(out_maps_tensor) * split)]

    val_in_maps_tensor = in_maps_tensor[int(len(in_maps_tensor) * split): len(in_maps_tensor)]
    val_out_maps_tensor = out_maps_tensor[int(len(out_maps_tensor) * split): len(out_maps_tensor)]
    

    tr_dataset = utils.data.TensorDataset(tr_in_maps_tensor, tr_out_maps_tensor)
    val_dataset = utils.data.TensorDataset(val_in_maps_tensor, val_out_maps_tensor)

    tr_dataloader = utils.data.DataLoader(
        tr_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True
        )

    val_dataloader = utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True
        )

    return tr_dataloader, val_dataloader

def preprocess(input_file, output_file, map_size, split=0.995, batch_size=8):

    inputs, outputs = read_data(inp_file=input_file, out_file=output_file)

    tr_dataloader, val_dataloader = create_iterators(
        inputs, 
        outputs, 
        map_size, 
        batch_size, 
        split
    )

    return tr_dataloader, val_dataloader