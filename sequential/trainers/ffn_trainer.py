from base.base_train import BaseTrain
from tqdm import tqdm
from copy import deepcopy

import numpy as np
import tensorflow as tf 

class FFNTrainer(BaseTrain): 
    def __init__(self, sess, model, data, config, logger): 
        super(FFNTrainer, self).__init__(sess, model, data, config, logger)
        self.test_it = 0
        self.dataset_it = 0
        self.data_list = [] 
        self.train_accuracy = [] 
        self.set_labels()
        self.store_data(self.data)
        self.all_test_accuracies = []


    def train_epoch(self): 
        loop = tqdm(range(self.config.num_iter_per_epoch))
        num_datasets = len(self.data_list)
        losses = [] 
        accs = [] 
        test_accuracies = [[] for x in range(num_datasets)] 

        for it in loop: 
            loss, acc = self.train_step()
            losses.append(loss)
            accs.append(acc)
            for idx in reversed(range(num_datasets)): 
                test_accuracy = self.test(self.data_list[idx])
                test_accuracies[idx].append(test_accuracy)
            
        loss = np.mean(losses)
        acc = np.mean(accs)
        self.train_accuracy.append(accs)

        if(len(test_accuracies) > 1):
            test_accuracies = np.mean(test_accuracies, axis=1)
        else: 
            test_accuracies = [np.mean(test_accuracies[0])]

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {} 
        summaries_dict[self.loss_label] = loss 
        summaries_dict[self.accuracy_label] = acc
        summaries_dict[self.test_accuracy_label] = test_accuracies[num_datasets-1]

        self.all_test_accuracies.append(test_accuracies) 

        for idx in reversed(range(num_datasets-1)): 
            test_accuracy_label = 'previous_acc_' + str(idx)
            summaries_dict[test_accuracy_label] = test_accuracies[idx]
        
        print('Iteration: %s' % self.test_it)
        print('Loss: %s' % loss)
        print('Training accuracy: %s' % acc)
        print('Current Test accuracy: %s' % test_accuracies[num_datasets-1])
        self.logger.summarize(self.test_it, summaries_dict=summaries_dict)
        self.test_it += 1

    def train_step(self): 
        batch_x, batch_y = self.data.train.next_batch(self.config.batch_size)
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: True}
        _, loss, acc = self.sess.run([self.model.train_step, self.model.cross_entropy, self.model.accuracy], feed_dict=feed_dict)

        return loss, acc 

    def test_previous_dataset(self, previous_data):
        self.test(previous_data)  

    def test(self, data): 
        batch_x, batch_y = data.test.next_batch(self.config.batch_size)
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: False}
        acc = self.sess.run(self.model.accuracy, feed_dict=feed_dict)

        return acc 

    def set_labels(self):
        self.loss_label = 'loss_' + str(self.dataset_it)
        self.accuracy_label = 'accuracy_' + str(self.dataset_it)
        self.test_accuracy_label = 'test_accuracy_' + str(self.dataset_it)

    def store_data(self, data): 
        data = deepcopy(data)
        self.data_list.append(data)

    def reset(self, data): 
        self.data = data
        self.store_data(self.data)
        self.test_it = 0  
        self.dataset_it += 1
        self.set_labels()
        global_tensor = self.model.init_global_step() 
        epoch_tensor = self.model.init_cur_epoch()
        init = tf.variables_initializer([global_tensor, epoch_tensor])
        self.sess.run(init)

        self.train_accuracy = []
        self.all_test_accuracies = [] 