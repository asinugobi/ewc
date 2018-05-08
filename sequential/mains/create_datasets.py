import tensorflow as tf
import numpy as np 
import sys 

sys.path.append('/home/asinugobi/tensorflow-1.5.0/tensorflow_pkg/ewc/sequential')

from data_loader.data_generator import DataGenerator
from data_loader.data_handler import DataHandler
from models.simple_cnn_model import SimpleCNNModel
from trainers.simple_cnn_trainer import SimpleCNNTrainer
from utils.config import process_config
from utils.plotting import *
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args
from scipy.io import savemat

def main():
    # capture the config path from the run arguments
    # then process the json configration file
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    # create tensorflow session
    sess = tf.Session()

    # create your data generator
    data = DataHandler(config)
    permuted_mnist = data.permute_mnist()
    permuted_mnist_2 = data.permute_mnist() 

    # save datasets 
    filename_1  = config.data_dir + 'permuted_mnist_1.pickle'
    filename_2  = config.data_dir + 'permuted_mnist_2.pickle'
    data.save_dataset(dataset=permuted_mnist, filename=filename_1)
    data.save_dataset(dataset=permuted_mnist_2, filename=filename_2)

if __name__ == '__main__':
    main()