import unittest
import tensorflow as tf 
import matplotlib.pyplot as plt
import sys 

sys.path.append('/home/asinugobi/tensorflow-1.5.0/tensorflow_pkg/ewc/sequential')

from data_loader.data_generator import DataGenerator
from data_loader.data_handler import DataHandler
from models.ffn_model import FFNModel
from trainers.ffn_trainer import FFNTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args
from utils.plotting import plot_results
from copy import deepcopy

class TestFFNEWC(unittest.TestCase):
    # Setup model for tests
    def setUp(self): 
        # capture the config path from the run arguments
        # then process the json configration file
        try:
            args = get_args()
            self.config = process_config(args.config)

        except:
            print("missing or invalid arguments")
            exit(0)
        
        create_dirs([self.config.summary_dir, self.config.checkpoint_dir])
        
        # MAKE SURE TO RESET GRAPH AFTER EACH TEST 
        tf.reset_default_graph() 
        
        self.sess = tf.Session() 
        self.model = FFNModel(self.config)
        self.data = DataHandler(self.config)
        self.mnist = self.data.get_dataset() 
        logger = Logger(self.sess, self.config)
        self.trainer = FFNTrainer(self.sess, self.model, self.mnist, self.config, logger)

    # def test_computing_fisher_matrix(self): 
    #     # train with normal SGD 
    #     self.trainer.train()

    #     # compute the fisher info matrix 
    #     variables = tf.trainable_variables()
    #     self.model.set_variable_list(variables)
 
    #     with self.sess.as_default(): 
    #         print('FM variables: ')
    #         for var in self.model.var_list:
    #             print(var.name)
    #             print(var.eval()) 
    #     print('==========')

    #     time = self.model.compute_fisher(self.mnist.validation.images, self.sess, num_samples=200, plot_diffs=True) 

        # print("Running time for computing FM: %s" % str(time))

    # def test_saving_previous_weights(self):
    #     # train with normal SGD 
    #     self.trainer.train() 

    #     variables = tf.trainable_variables() 
    #     self.model.set_variable_list(variables)

    #     # save the variables 
    #     self.model.star(self.sess)

    #     variables_eval = [] 
    #     with self.sess.as_default(): 
    #         for v in self.model.var_list: 
    #             variables_eval.append(v.eval())

    #         for i in range(len(variables_eval)): 
    #             print('Pre-saved: variables: ')
    #             print(variables_eval[i])
    #             print('Saved variables: ')
    #             print(self.model.star_vars[i])
    
    def test_transfer_ewc(self):
        # train with normal SGD 
        self.trainer.train() 

        # save the variables 
        variables = tf.trainable_variables()
        self.model.set_variable_list(variables)
        self.model.star(self.sess)

        # compute fisher matrix 
        time = self.model.compute_fisher(self.mnist.validation.images, self.sess, num_samples=200, plot_diffs=False) 
        print("Running time for computing FM: %s" % str(time))

        # set loss/optimizer 
        self.model.set_ewc_loss(lam=15)
        self.model.reset_train_step(loss=self.model.ewc_loss, variables=variables)

        # reset paramaters for training on new data 
        permutated_mnist_2 = self.data.permute_mnist() 
        self.trainer.reset(permutated_mnist_2)

        # train on new dataset 
        self.trainer.train()
        
        # plot results 
        test_plots = [[] for x in range(2)]
        for idx in range(self.config.num_epochs+1):
            test_plots[0].append(self.trainer.all_test_accuracies[idx][0])
            test_plots[1].append(self.trainer.all_test_accuracies[idx][1])

        plot_results(num_iterations=self.config.num_epochs+1, train_plots=self.trainer.train_accuracy, test_plots=test_plots)

if __name__ == '__main__': 
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFFNEWC)
    unittest.TextTestRunner(verbosity=2).run(suite)






