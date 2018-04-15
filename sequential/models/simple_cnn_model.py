from base.base_model import BaseModel
from networks.simple_cnn import SimpleCNN
import tensorflow as tf

class SimpleCNNModel(BaseModel):
    def __init__(self, config):
        super(SimpleCNNModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)

        # define placeholder for x and y 
        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.y = tf.placeholder(tf.float32, shape=[None, 10])
        self.phase = tf.placeholder(tf.bool, name='phase')
        self.dropout = tf.placeholder(dtype=tf.float32, name='dropout')

        # network architecture (simple 4-layer cnn)
        self.cnn = SimpleCNN(input=self.x,
                        phase=self.phase,
                        dropout=self.dropout,
                        num_classes=self.config.num_classes,
                        skip_layer=[''],
                        weights_path='DEFAULT')
        self.cnn.create_graph() 

        # define loss 
        self.loss() 

        # define optimizer 
        self.optimize() 
                                                                
        # performance metrics 
        self.set_metrics() 

    def loss(self): 
        # loss definition 
        with tf.name_scope('loss'):
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.cnn.sigma))

    def optimize(self):
        # NOTE: when training, the moving_mean and moving_variance need to be updated. By default the update ops are placed in tf.GraphKeys.UPDATE_OPS, so they need to be added as a dependency to the train_op.(https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        with tf.control_dependencies(update_ops), tf.name_scope('optimize'): 
            learning_rate = self.config.learning_rate
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            self.train_step = optimizer.minimize(self.cross_entropy)

    def set_metrics(self): 
        with tf.name_scope('accuracy'): 
                    correct_prediction = tf.equal(tf.argmax(self.cnn.sigma, 1), tf.argmax(self.y, 1))
                    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def init_saver(self):
        # here you initalize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def star(self):
        # used for saving optimal weights after most recent task training
        self.star_vars = []

        for v in range(len(self.var_list)):
            self.star_vars.append(self.var_list[v].eval())

    def restore(self, sess):
        # reassign optimal weights for latest task
        if hasattr(self, "star_vars"):
            for v in range(len(self.var_list)):
                sess.run(self.var_list[v].assign(self.star_vars[v]))

    def set_vanilla_loss(self):
        self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.cross_entropy)

    def update_ewc_loss(self, lam):
        # elastic weight consolidation
        # lam is weighting for previous task(s) constraints

        if not hasattr(self, "ewc_loss"):
            self.ewc_loss = self.cross_entropy

        for v in range(len(self.var_list)):
            self.ewc_loss += (lam/2) * tf.reduce_sum(tf.multiply(self.F_accum[v].astype(np.float32),tf.square(self.var_list[v] - self.star_vars[v])))
        self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.ewc_loss)

