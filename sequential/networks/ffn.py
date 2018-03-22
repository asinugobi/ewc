import tensorflow as tf
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from IPython import display

# variable initialization functions
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

class FeedForwardNet(object):
    def __init__(self, x, y_):
        self.in_dim = int(x.get_shape()[1]) # 784 for MNIST
        self.out_dim = int(y_.get_shape()[1]) # 10 for MNIST
        self.x = x # input placeholder

    def create_graph(self): 
        # simple 2-layer network
        W1 = weight_variable([self.in_dim, 50])
        b1 = bias_variable([50])

        W2 = weight_variable([50, self.out_dim])
        b2 = bias_variable([self.out_dim])

        h1 = tf.nn.relu(tf.matmul(self.x, W1) + b1) # hidden layer
        h1 = tf.nn.dropout(h1, 0.8)
        self.y = tf.matmul(h1, W2) + b2 # output layer

        self.var_list = [W1, b1, W2, b2]

    


