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
        self.sess = tf.Session() 
        self.model = FFNModel(config)
        data = DataHandler(config)
        mnist = data.get_dataset() 
        logger = Logger(self.sess, config)
        self.trainer = FFNTrainer(self.sess, self.model, mnist, config, logger)


    def test_save_restore_all(self): 
        self.trainer.train()
        variables = tf.trainable_variables()  
        self.model.save(self.sess) 

        # Print out all variables 
        for var in variables: 
            print(var.name)
        
        self.model.load(self.sess) 

        assertEquals(variables, tf.trainable_variables(), 'not the same variables')

if __name__ == '__main__': 
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFFN)
    unittest.TextTestRunner(verbosity=2).run(suite)






