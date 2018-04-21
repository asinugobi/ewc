import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time 

from base.base_model import BaseModel
from networks.ffn import FeedForwardNet
from copy import deepcopy

class FFNModel(BaseModel):
    def __init__(self, config):
        super(FFNModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)

        # define placeholder for x and y 
        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.y = tf.placeholder(tf.float32, shape=[None, 10])

        # network architecture (simple 2-layer ffn)
        self.net = FeedForwardNet(self.x, self.y)
        self.net.create_graph() 

        # define loss 
        self.loss() 

        # define optimizer 
        self.optimize() 
                                                                
        # performance metrics 
        self.set_metrics() 

    def loss(self): 
        # loss definition 
        with tf.name_scope('loss'):
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.net.y))

    def optimize(self):
        # NOTE: when training, the moving_mean and moving_variance need to be updated. By default the update ops are placed in tf.GraphKeys.UPDATE_OPS, so they need to be added as a dependency to the train_op.(https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        with tf.control_dependencies(update_ops), tf.name_scope('optimize'): 
            learning_rate = 0.1 
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            self.train_step = self.optimizer.minimize(self.cross_entropy)

    def reset_train_step(self, loss, variables=tf.trainable_variables()):
        self.train_step = self.optimizer.minimize(loss,                                           var_list=variables)

    def set_metrics(self): 
        with tf.name_scope('accuracy'): 
                    correct_prediction = tf.equal(tf.argmax(self.net.y, 1), tf.argmax(self.y, 1))
                    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def init_saver(self):
        # here you initalize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def reset_saver(self, vars): 
        self.saver = tf.train.Saver(var_list=vars, max_to_keep=self.config.max_to_keep)

    def star(self, sess):
        # used for saving optimal weights after most recent task training
        self.star_vars = []

        with sess.as_default(): 
            for v in range(len(self.var_list)):
                self.star_vars.append(self.var_list[v].eval())

    def restore(self, sess):
        # reassign optimal weights for latest task
        if hasattr(self, "star_vars"):
            for v in range(len(self.var_list)):
                sess.run(self.var_list[v].assign(self.star_vars[v]))

    def set_variable_list(self, var_list): 
        self.var_list = var_list 

    def set_ewc_loss(self, lam):
        # elastic weight consolidation
        # lam is weighting for previous task(s) constraints

        if not hasattr(self, "ewc_loss"):
            self.ewc_loss = self.cross_entropy

        for v in range(len(self.var_list)):
            self.ewc_loss += (lam/2) * tf.reduce_sum(tf.multiply(self.F_accum[v].astype(np.float32),tf.square(self.var_list[v] - self.star_vars[v])))

    def compute_fisher(self, imgset, sess, num_samples=200, plot_diffs=False, disp_freq=10):
        # computer Fisher information for each parameter

        # initialize Fisher information for most recent task
        self.F_accum = []
        for v in range(len(self.var_list)):
            self.F_accum.append(np.zeros(self.var_list[v].get_shape().as_list()))

        # sampling a random class from softmax
        probs = tf.nn.softmax(self.net.y)
        class_ind = tf.to_int32(tf.multinomial(tf.log(probs), 1)[0][0])

        if(plot_diffs):
            # track differences in mean Fisher info
            F_prev = deepcopy(self.F_accum)
            mean_diffs = np.zeros(0)

        iterations = [] 
        mean_diffs = [] 
        start = time.time()
        for i in range(num_samples):
            # select random input image
            im_ind = np.random.randint(imgset.shape[0])

            # compute first-order derivatives
            ders = sess.run(tf.gradients(tf.log(probs[0,class_ind]), self.var_list), feed_dict={self.x: imgset[im_ind:im_ind+1]})
            # square the derivatives and add to total
            for v in range(len(self.F_accum)):
                self.F_accum[v] += np.square(ders[v])
            if(plot_diffs):
                if i % disp_freq == 0 and i > 0:
                    # recording mean diffs of F
                    iterations.append(i)
                    F_diff = 0

                    for v in range(len(self.F_accum)):
                        F_diff += np.sum(np.absolute(self.F_accum[v]/(i+1) - F_prev[v]))
                    mean_diff = np.mean(F_diff)
                    mean_diffs = np.append(mean_diffs, mean_diff)
                    
                    # update the previous Fisher matrix 
                    for v in range(len(self.F_accum)):
                        F_prev[v] = self.F_accum[v]/(i+1)

        # divide totals by number of samples
        for v in range(len(self.F_accum)):
            self.F_accum[v] /= num_samples

        end = time.time() 

        # plot results
        if(plot_diffs):
            plt.plot(iterations, mean_diffs)
            plt.xlabel("Number of samples")
            plt.ylabel("Mean absolute Fisher difference")
            plt.show()

        return end - start 


