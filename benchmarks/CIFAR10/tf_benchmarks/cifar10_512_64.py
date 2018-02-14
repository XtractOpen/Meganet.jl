"""Runs benchmark using tensorflow and 1 cpu thread
   Runs 1 epoch with 512/103 Train/Val split in 8.56s
"""
from utils_input import *
from Resnet import ResnetModel

DATA_DIR = "path/to/where/you/want/your/data/stored"

if __name__ == '__main__':
    # Load Data
    ntrain = 512
    nval = 103
    train_data, valid_data = load_cifar10(DATA_DIR, ntrain, nval) # Note this does not evenly distribute classes

    # Set config 
    modelConfig = {
        "channels": [16,32,64,64],
        "batchSize": 64,
        "numUnits": 3,
        "numBlocksPerUnit": [3,3,3],
        "h": [1.,1.,1.],
        "maxEpochs": 1,
        "kSize": 3,
        "xTrain": train_data[0],
        "xValid": valid_data[0],
        "yTrain": train_data[1],
        "yValid": valid_data[1],
    }

    # Train model
    model = ResnetModel(modelConfig)
    model.train()
