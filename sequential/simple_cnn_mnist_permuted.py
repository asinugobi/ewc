import tensorflow as tf

from data_loader.data_generator import DataGenerator
from data_loader.data_handler import DataHandler
from models.simple_cnn_model import SimpleCNNModel
from trainers.simple_cnn_trainer import SimpleCNNTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args


def main():
    # capture the config path from the run arguments
    # then process the json configration file
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    
    # create tensorflow session
    sess = tf.Session()
    
    # create instance of the model you want
    model = SimpleCNNModel(config)
    
    # create your data generator
    data = DataHandler(config)
    data.permute_mnist() 
    permutated_mnist = data.get_dataset()
    
    # create tensorboard logger
    logger = Logger(sess, config)
    
    # create trainer and path all previous components to it
    trainer = SimpleCNNTrainer(sess, model, permutated_mnist, config, logger)

    # here you train your model
    trainer.train()



if __name__ == '__main__':
    main()
