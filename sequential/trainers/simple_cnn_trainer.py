from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
import tensorflow as tf 

class SimpleCNNTrainer(BaseTrain): 
    def __init__(self, sess, model, data, config, logger): 
        super(SimpleCNNTrainer, self).__init__(sess, model, data, config, logger)
        self.test_it = 0
        self.dataset_it = 0
        self.set_labels()

    def train_epoch(self): 
        loop = tqdm(range(self.config.num_iter_per_epoch))
        losses = [] 
        accs = [] 
        test_accuracies = [] 

        for it in loop: 
            loss, acc = self.train_step()
            test_accuracy = self.test()
            losses.append(loss)
            accs.append(acc)
            test_accuracies.append(test_accuracy)

        loss = np.mean(losses)
        acc = np.mean(accs)
        test_accuracy = np.mean(test_accuracies)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {} 
        summaries_dict[self.loss_label] = loss 
        summaries_dict[self.accuracy_label] = acc 
        summaries_dict[self.test_accuracy_label] = test_accuracy

        # Print results. 
        print('Iteration: %s' % self.test_it)
        print('Loss: %s' % loss)
        print('Training accuracy: %s' % acc)
        print('Test accuracy: %s' % test_accuracy)

        self.logger.summarize(self.test_it, summaries_dict=summaries_dict)
        self.test_it += 1

    def train_step(self): 
        batch_x, batch_y = self.data.train.next_batch(self.config.batch_size)
        feed_dict = {self.model.x: batch_x, 
                     self.model.y: batch_y, 
                     self.model.phase: 1,
                     self.model.dropout: self.config.dropout}
        _, loss, acc = self.sess.run([self.model.train_step,
                                      self.model.cross_entropy, self.model.accuracy], feed_dict=feed_dict)
        return loss, acc 

    def test(self): 
        batch_x, batch_y = self.data.test.next_batch(self.config.batch_size)
        feed_dict = {self.model.x: batch_x, 
                     self.model.y: batch_y, 
                     self.model.phase: 0,
                     self.model.dropout: self.config.dropout}
        acc = self.sess.run(self.model.accuracy, feed_dict=feed_dict)

        return acc 

    def set_labels(self):
        self.loss_label = 'loss_' + str(self.dataset_it)
        self.accuracy_label = 'accuracy_' + str(self.dataset_it)
        self.test_accuracy_label = 'test_accuracy_' + str(self.dataset_it)

    def reset(self, data): 
        self.data = data
        self.test_it = 0  
        self.dataset_it += 1
        self.set_labels()
        global_tensor = self.model.init_global_step() 
        epoch_tensor = self.model.init_cur_epoch()
        init = tf.variables_initializer([global_tensor, epoch_tensor])
        self.sess.run(init)
