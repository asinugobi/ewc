#!/bin/bash
# Shell script for running transfer ewc experiments 

# 1. Run EWC on all network layers 
python simple_cnn_varying_ewc.py -c ../configs/simple_cnn_mnist_permuted_transfer_ewc.json 

# 2. Run EWC on second conv layer only  
python simple_cnn_layerwise_ewc.py -c ../configs/simple_cnn_layerwise_4.json

# 3. Run EWC on both conv layers
python simple_cnn_layerwise_ewc.py -c ../configs/simple_cnn_layerwise_8.json
 