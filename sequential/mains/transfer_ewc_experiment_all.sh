#!/bin/bash
# Shell script for running transfer ewc experiments 

# Experiment 1: Baseline (no ewc; transfer both conv layers)
# python simple_cnn_layerwise_ewc.py -c ../configs/simple_cnn_layerwise_0.json

# Experiment 2: EWC on second conv layer only  
python simple_cnn_layerwise_ewc.py -c ../configs/simple_cnn_layerwise_4.json

# Experiment 3: EWC on both conv layers only 
python simple_cnn_layerwise_ewc.py -c ../configs/simple_cnn_layerwise_8.json

# Experiment 4: EWC on second conv layer + fully-connected layers 
python simple_cnn_layerwise_ewc.py -c ../configs/simple_cnn_layerwise_4b.json

# Experiment 5: EWC on both conv layers + fully-connected layers 
python simple_cnn_layerwise_ewc.py -c ../configs/simple_cnn_layerwise_8b.json

# Experiment 6: EWC on fully-connected layers only 
python simple_cnn_layerwise_ewc.py -c ../configs/simple_cnn_layerwise_0b.json