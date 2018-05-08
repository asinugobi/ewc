import tensorflow as tf
import numpy as np 
import os.path 
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

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
 
    # create tensorflow session
    sess = tf.Session()
 
    # create instance of the model you want
    model = SimpleCNNModel(config)
 
    # create your data generator
    data = DataHandler(config)

    # create source and target tasks
    filename_1 = '../data/experiments/permuted_mnist_1.pickle'
    filename_2 = '../data/experiments/permuted_mnist_2.pickle' 
    if not os.path.isfile(filename_1): 
        permuted_mnist = data.permute_mnist()
        data.save_dataset(dataset=permuted_mnist, filename=filename_1)
    else: 
        permuted_mnist = data.load_dataset(filename=filename_1)
        print('check: p1')

    if not os.path.isfile(filename_2): 
        permuted_mnist_2 = data.permute_mnist()
        data.save_dataset(dataset=permuted_mnist_2, filename=filename_2)
    else: 
        permuted_mnist_2 = data.load_dataset(filename=filename_2)
        print('check: p2')
    
    # create tensorboard logger
    logger = Logger(sess, config)
 
    # create trainer and path all previous components to it
    trainer = SimpleCNNTrainer(sess, model, permuted_mnist, config, logger)

    # train your model
    # variables = tf.trainable_variables() 
    # print('length of variables: %s' % len(variables))
    # print(variables)
    trainer.train()
    model.save(sess)

    # save weights to be transferred 
    variables = tf.trainable_variables() 
    # print('length of variables: %s' % len(variables))

    frozen_variables = variables[:config.nth_layer] 
    transferred_variables = variables[config.nth_layer : config.top_layers]
    top_layer_variables = variables[config.top_layers:]
    trainable_variables = transferred_variables + top_layer_variables

    # model.reset_saver(saved_variables)
    # model.save(sess)

    ######################################################################################## TRANSFER TO NEW DATASET ###################
    ##################################################################

    ## transfer weights to new model
    # save the transferred variables to model 
    if config.ewc_on_top_layers:
        model.set_variable_list(trainable_variables)
    else:  
        model.set_variable_list(transferred_variables)

    model.star(sess) 

    # # freeze weights (train only reinitialized variables)
    # model.reset_train_step(variables=reinitialized_variables)

    # compute fisher matrix 
    time = model.compute_fisher(permutated_mnist.validation.images, sess, num_samples=200, plot_diffs=False) 
    print("Running time for computing FM: %s" % str(time))

    # cycle through ewc penalties 
    min = 0
    max = 201
    step = 20
    if not (config.title == 'Experiment 1'): 
        ewc_penalty = range(min, max, step)
    else: 
        ewc_penalty = [0]

    # reset paramaters for training on new data 
    average_losses = [] 
    path = config.path

    # dictionaries for saving data 
    save_test_data = {} 
    save_loss_data = {}
    save_to_mat = {}  
    
    for penalty in ewc_penalty: 
        print('Evaluating EWC Penalty: %s' % str(penalty))
        print('====================')

        # restore original model 
        model.load(sess)

        # reinitialize top layer weights 
        if not config.ewc_on_top_layers: 
            init = tf.variables_initializer(top_layer_variables)
            sess.run(init)
    
        # set loss/optimizer 
        model.reset_ewc_loss() 
        model.set_ewc_loss(lam=penalty)
        model.reset_train_step(loss=model.ewc_loss, variables=trainable_variables)

        # train on new dataset 
        trainer.reset(permuted_mnist_2)
        trainer.train()
        
        # plot results 
        test_plots = [[] for x in range(2)]
        for idx in range(config.num_epochs+1):
            test_plots[0].append(trainer.all_test_accuracies[idx][0])
            test_plots[1].append(trainer.all_test_accuracies[idx][1])

        loss_plots = [[] for x in range(2)]
        for idx in range(config.num_epochs+1):
            loss_plots[0].append(trainer.all_test_losses[idx][0])
            loss_plots[1].append(trainer.all_test_losses[idx][1])

        average_losses.append(np.mean(loss_plots[0]))

        add_title = ' (penalty = ' + str(penalty) + ')'

        # plot results
        plot_results(num_iterations=config.num_epochs+1, train_plots=trainer.train_accuracy, test_plots=test_plots, loss_plots=loss_plots, save=True, show=False, path=path, experiment='simple_cnn_ewc_' + str(penalty), title=config.title + add_title)

        # save test and loss results 
        test_data_key = config.exp_name + '_test_' + str(penalty)
        loss_data_key = config.exp_name + '_loss_' + str(penalty)
        save_test_data[test_data_key] = test_plots
        save_loss_data[loss_data_key] = loss_plots 
        

    # plot average loss versus ewc penalty 
    plot_varying_penalty(penalties=ewc_penalty, average_loss=average_losses, path=path, experiment='simple_cnn_ewc_', save=True, title=config.title) 

    # save all results in general dictionary 
    average_losses_data_key = config.exp_name + 'avg_loss' 
    save_to_mat[average_losses_data_key] = average_losses
    save_to_mat['test_accuracies'] = save_test_data
    save_to_mat['loss'] = save_loss_data

    # save dictonary to matlab file 
    savemat(config.data_dir, save_to_mat)

if __name__ == '__main__':
    main()
