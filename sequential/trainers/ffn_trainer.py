from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np

class FFNTrainer(BaseTrain): 
    def __init__(self, sess, model, data, config, logger): 
        super(FFNTrainer, self).__init__(sess, model, data, config, logger)
        self.test_it = 0

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
        summaries_dict['loss'] = loss 
        summaries_dict['acc'] = acc 
        summaries_dict['test_accuracy'] = test_accuracy
        print('Iteration: %s' % test_it)
        print('Loss: %s' % loss)
        print('Training accuracy: %s' % acc)
        print('Test accuracy: %s' % test_accuracy)
        self.logger.summarize(self.test_it, summaries_dict=summaries_dict)
        self.test_it += 1

    def train_step(self): 
        batch_x, batch_y = self.data.train.next_batch(self.config.batch_size)
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: True}
        _, loss, acc = self.sess.run([self.model.train_step, self.model.cross_entropy, self.model.accuracy], feed_dict=feed_dict)

        return loss, acc 

    def test(self): 
        batch_x, batch_y = self.data.test.next_batch(self.config.batch_size)
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: False}
        acc = self.sess.run(self.model.accuracy, feed_dict=feed_dict)

        return acc 
