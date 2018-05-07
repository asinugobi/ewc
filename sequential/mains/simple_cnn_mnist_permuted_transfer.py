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
    permutated_mnist = data.permute_mnist()
    
    # create tensorboard logger
    logger = Logger(sess, config)
 
    # create trainer and path all previous components to it
    trainer = SimpleCNNTrainer(sess, model, permutated_mnist, config, logger)

    # train your model
    trainer.train()

    # plot_results(num_iterations=config.num_epochs+1, train_plots=trainer.train_accuracy, test_plots=[np.mean(trainer.all_test_accuracies, axis=1)])
    test_plots = [[] for x in range(1)]
    for idx in range(config.num_epochs+1):
        test_plots[0].append(trainer.all_test_accuracies[idx][0])

    loss_plots = [[] for x in range(1)]
    for idx in range(config.num_epochs+1):
        loss_plots[0].append(trainer.all_test_losses[idx][0])

    plot_results(num_iterations=config.num_epochs+1, train_plots=trainer.train_accuracy, test_plots=test_plots, loss_plots=loss_plots, save=True, show=False, path=config.path, experiment='simple_cnn_initial')

    # save weights to be transferred 
    # variables = tf.trainable_variables() 
    # print('length of variables: %s' % len(variables))
    # # for v in variables: 
    # #     print(v.name)
    
    # n = 8
    # saved_variables = variables[:n] 
    # reinitialized_variables = variables[n:] 
    # model.reset_saver(saved_variables)
    # model.save(sess)

    ##################################################################
    ##################### TRANSFER TO NEW DATASET ####################
    ##################################################################

    ## transfer weights to new model
    # reinitialize top layer weights 
    # init = tf.variables_initializer(reinitialized_variables)
    # sess.run(init)

    # reload saved weights 
    # model.load(sess)

    # freeze weights (train only reinitialized variables)
    # model.reset_train_step(variables=reinitialized_variables)

    # reset paramaters for training on new data 
    permutated_mnist_2 = data.permute_mnist() 
    trainer.reset(permutated_mnist_2)

    # save weights to be transferred 
    variables = tf.trainable_variables() 
    print('length of variables: %s' % len(variables))

    frozen_variables = variables[:config.nth_layer] 
    transferred_variables = variables[config.nth_layer : config.top_layers]
    top_layer_variables = variables[config.top_layers:]
    trainable_variables = transferred_variables + top_layer_variables

    # reinitialize top layer weights 
    init = tf.variables_initializer(top_layer_variables)
    sess.run(init)

    model.reset_train_step(loss=model.cross_entropy, variables=trainable_variables)

    # train on new dataset 
    trainer.train()
    test_plots = [[] for x in range(2)]
    for idx in range(config.num_epochs+1):
        test_plots[0].append(trainer.all_test_accuracies[idx][0])
        test_plots[1].append(trainer.all_test_accuracies[idx][1])

    loss_plots = [[] for x in range(2)]
    for idx in range(config.num_epochs+1):
        loss_plots[0].append(trainer.all_test_losses[idx][0])
        loss_plots[1].append(trainer.all_test_losses[idx][1])

    plot_results(num_iterations=config.num_epochs+1, train_plots=trainer.train_accuracy, test_plots=test_plots, loss_plots=loss_plots, save=True, show=False, path=config.path, experiment='simple_cnn_transfer_baseline_2') 

    ######################################################################################## TRANSFER TO NEW DATASET ###################
    ##################################################################

    # reset paramaters for training on new data 
    # permutated_mnist_3 = data.permute_mnist() 
    # trainer.reset(permutated_mnist_3)

    # # train on new dataset 
    # trainer.train()
    # test_plots = [[] for x in range(3)]
    # for idx in range(config.num_epochs+1):
    #     test_plots[0].append(trainer.all_test_accuracies[idx][0])
    #     test_plots[1].append(trainer.all_test_accuracies[idx][1])
    #     test_plots[2].append(trainer.all_test_accuracies[idx][2])
        
    # plot_results(num_iterations=config.num_epochs+1, train_plots=trainer.train_accuracy, test_plots=test_plots)

if __name__ == '__main__':
    main()
