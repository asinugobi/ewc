import tensorflow as tf
import numpy as np 
import sys 

sys.path.append('/home/asinugobi/tensorflow-1.5.0/tensorflow_pkg/ewc/sequential')

from data_loader.data_generator import DataGenerator
from data_loader.data_handler import DataHandler
from models.ffn_model import FFNModel
from trainers.ffn_trainer import FFNTrainer
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
    model.save(sess)

        # save weights to be transferred (TO-DO)
    # variables = tf.trainable_variables() 
    # print('length of variables: %s' % len(variables))
    
    # n = 2
    # saved_variables = variables[:n] 
    # reinitialized_variables = variables[n:] 
    # model.reset_saver(saved_variables)
    # model.save(sess)

    ######################################################################################## TRANSFER TO NEW DATASET ###################
    ##################################################################

    # ## transfer weights to new model 
    # # reinitialize top layer weights 
    # init = tf.variables_initializer(reinitialized_variables)
    # sess.run(init)

    # # reload saved weights 
    # model.load(sess)

    # # freeze weights (train only reinitialized variables)
    # model.reset_train_step(variables=reinitialized_variables)

    # save the variables 
    variables = tf.trainable_variables()
    model.set_variable_list(variables)
    model.star(sess)

    # compute fisher matrix 
    time = model.compute_fisher(permutated_mnist.validation.images, sess, num_samples=200, plot_diffs=False) 
    print("Running time for computing FM: %s" % str(time))

    # cycle through ewc penalties 
    min = 0
    max = 201
    step = 20
    # ewc_penalty = range(min, max, step)
    ewc_penalty = [0, 15]

    # reset paramaters for training on new data 
    permutated_mnist_2 = data.permute_mnist() 
    average_losses = [] 
    
    for penalty in ewc_penalty: 
        # restore original model 
        model.load(sess)

        # set loss/optimizer 
        model.set_ewc_loss(lam=penalty)
        model.reset_train_step(loss=model.ewc_loss, variables=variables)

        # train on new dataset 
        trainer.reset(permutated_mnist_2)
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

        # plot_results(num_iterations=config.num_epochs+1, train_plots=trainer.train_accuracy, test_plots=test_plots, loss_plots=loss_plots, save=True, show=False, path='../figures/', experiment='testing' + str(penalty))

    plot_varying_penalty(penalties=ewc_penalty, average_loss=average_losses)

         

if __name__ == '__main__':
    main()
