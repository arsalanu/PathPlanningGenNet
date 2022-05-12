## A generative neural network approach for single-shot autonomous robot path planning

This is the neural network implementation for the design outlined in my MEng paper "Investigating a generative neural network approach for single-shot autonomous robot path planning." It includes the source files used to generate the results, create the dataset, complete training, and evaluate the model. It also includes the final generated model used to run inference described in the paper.

The method is implemented in Python 3, using PyTorch.


`./data` will include the input and output CSVs of the maps used for training, each row contains 4096 values which are split into 64x64 maps in preprocessing, once generated.

`./examples/` includes PNG examples of inference outputs generated from evaluation.

`./network/` includes the PyTorch generated `.pth` saved model checkpoint. This can be loaded in to run inference with the last saved weights.

****
#### Dataset generation

The procedural generation script includes the optimal Dijkstra's algorithm and parameters used to generate a random obstacle map which constitutes the inputs for the training dataset. This can be run using:

```python3 ground_truth_generator.py {plot/none} {map size} {obstacle number} {obstacle limit}```


****
#### Training the model

Once the data is generated, the network can simply be trained from outside of the project folder by running:

```python3 PathPlanning_Network```

Parameters can be modified from inside `training.py`, `main.py`, `postproc.py` and `preproc.py` to change the network functionality.


