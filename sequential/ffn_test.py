import unittest
import tensorflow as tf 

from data_loader.data_generator import DataGenerator
from data_loader.data_handler import DataHandler
from models.ffn_model import FFNModel
from trainers.ffn_trainer import FFNTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args
from copy import deepcopy

class TestFFN(unittest.TestCase):
    # Setup model for tests
    def setUp(self): 
        # capture the config path from the run arguments
        # then process the json configration file
        try:
            args = get_args()
            config = process_config(args.config)

        except:
            print("missing or invalid arguments")
            exit(0)
        
        create_dirs([config.summary_dir, config.checkpoint_dir])
        
        # MAKE SURE TO RESET GRAPH AFTER EACH TEST 
        tf.reset_default_graph() 
        
        self.sess = tf.Session() 
        self.model = FFNModel(config)
        data = DataHandler(config)
        mnist = data.get_dataset() 
        logger = Logger(self.sess, config)
        self.trainer = FFNTrainer(self.sess, self.model, mnist, config, logger)

    # def test_reinitialize_variables(self):
    #     self.trainer.train()
    #     variables = tf.trainable_variables() 

    #     # Print out all variables
    #     with self.sess.as_default(): 
    #         for var in variables: 
    #             print(var.name)
    #             print(var.eval())

    #     # Reinitialize all variables 
    #     init = tf.variables_initializer(variables)
    #     self.sess.run(init)
    #     reinitialized_variables = tf.trainable_variables() 

    #     with self.sess.as_default(): 
    #         for var in reinitialized_variables:
    #             print(var.name)
    #             print(var.eval()) 
        

    # def test_save_restore_all(self): 
    #     self.trainer.train()
    #     variables = tf.trainable_variables()  
    #     self.model.save(self.sess) 

    #     with self.sess.as_default(): 
    #         print('SAVED VARIABLES: ')
    #         for var in variables:
    #             print(var.name)
    #             print(var.eval()) 
    #     print('==========')
        
    #     # Reinitialize variables 
    #     init = tf.variables_initializer(variables)
    #     self.sess.run(init)

    #     # Print reinitialized variables 
    #     with self.sess.as_default(): 
    #         print('REINITIALIZED VARIABLES: ')
    #         for var in variables:
    #             print(var.name)
    #             print(var.eval()) 
    #     print('==========')

    #     self.model.load(self.sess) 
    #     with self.sess.as_default(): 
    #         print('LOADED VARIABLES: ')
    #         for var in variables:
    #             print(var.name)
    #             print(var.eval()) 

    def test_save_restore_part(self): 
        self.trainer.train()
        variables = tf.trainable_variables()  
        saved_variables = variables[1:]
        reinitalized_variables = [variables[0]]
        self.model.reset_saver(saved_variables)

        self.model.save(self.sess) 
        with self.sess.as_default(): 
            print('BEFORE REINITIALIZATION: ')
            for var in variables:
                print(var.name)
                print(var.eval()) 
        print('==========')
        
        # Reinitialize the first variable only
        init = tf.variables_initializer(reinitalized_variables)
        self.sess.run(init)

        # Print reinitialized variables 
        with self.sess.as_default(): 
            print('AFTER REINITIALIZATION: ')
            for var in variables:
                print(var.name)
                print(var.eval()) 
        print('==========')

        self.model.load(self.sess) 
        with self.sess.as_default(): 
            print('AFTER LOADING: ')
            for var in variables:
                print(var.name)
                print(var.eval()) 

        # self.assertEqual(variables, tf.trainable_variables(), 'not the same variables')

if __name__ == '__main__': 
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFFN)
    unittest.TextTestRunner(verbosity=2).run(suite)






