import tensorflow as tf

from data_loader.data_generator import DataGenerator
from data_loader.data_handler import DataHandler
from models.ffn_model import FFNModel
from trainers.ffn_trainer import FFNTrainer
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
    model = FFNModel(config)
 
    # create your data generator
    data = DataHandler(config)
    permutated_mnist = data.permute_mnist()
    
    # create tensorboard logger
    logger = Logger(sess, config)
 
    # create trainer and path all previous components to it
    trainer = FFNTrainer(sess, model, permutated_mnist, config, logger)

    # here you train your model
    trainer.train()

    # save weights to be transferred (TO-DO)
    model.reset_saver(save)
    model.save(sess)

    ##################################################################
    ##################### TRANSFER TO NEW DATASET ####################
    ##################################################################

    # reset graph 
    tf.reset_default_graph() 

    # create new dataset for model 
    permutated_mnist_2 = data.permute_mnist() 

    # transfer weights to new model (TODO)
    model.load(sess)



if __name__ == '__main__':
    main()
