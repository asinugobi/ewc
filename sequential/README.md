# Transfer learning: Sequential 

This module focuses on applying layer-wise elastic consolidation to a convolutional neural network (CNN) in a sequential learning setting. 

## Transfer between Permuted MNIST 

In this module, we focus on transferring from one permuted MNIST dataset to another. We apply elastic weight consolidation to different subsets of the CNN to test whether layer-wise EWC can boost forward-transfer performance and/or mitigate catastrophic interference.  

### Running code 
To run all experiments simply enter the `mains` directory and the following command in your terminal: `./transfer_ewc_experiment_all.sh`

Note that you can comment and uncomment the experiments you would like to run in the bash script.  