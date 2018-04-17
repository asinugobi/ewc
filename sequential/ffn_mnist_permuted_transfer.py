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

    # train your model
    trainer.train()

    # save weights to be transferred (TO-DO)
    variables = tf.trainable_variables() 
    print('length of variables: %s' % len(variables))
    
    n = 2
    saved_variables = variables[:n] 
    reinitialized_variables = variables[n:] 
    model.reset_saver(saved_variables)
    model.save(sess)

    ##################################################################
    ##################### TRANSFER TO NEW DATASET ####################
    ##################################################################

    ## transfer weights to new model (TODO)
    # reinitialize top layer weights 
    init = tf.variables_initializer(reinitialized_variables)
    sess.run(init)

    # reload saved weights 
    model.load(sess)

    # freeze weights (train only reinitialized variables)
    model.reset_train_step(variables=reinitialized_variables)

    # reset paramaters for training on new data 
    permutated_mnist_2 = data.permute_mnist() 
    trainer.reset(permutated_mnist_2)

    # train on new dataset 
    trainer.train()




if __name__ == '__main__':
    main()
