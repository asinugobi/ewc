#!/bin/bash
# Shell script for running transfer ewc experiments 

# 1. Run EWC on fully-connected layers only 
# python simple_cnn_layerwise_ewc.py -c ../configs/simple_cnn_layerwise_0b.json

# 2. Run EWC on second conv layer + fully-connected layers 
# python simple_cnn_layerwise_ewc.py -c ../configs/simple_cnn_layerwise_4b.json

# 3. Run EWC on both conv layers + fully-connected layers 
python simple_cnn_layerwise_ewc.py -c ../configs/simple_cnn_layerwise_8b.json
