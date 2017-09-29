import numpy as np
import tensorflow as tf

class Model():
    def __init__(self, dataHandler=None):
        self.dh = dataHandler


    def cnn_forDepthMaps(self, input):
        """Model function for CNN."""
        input_layer = tf.reshape(input, [-1, 54, 96, 1])

        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=10,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        return conv1


    def dense_layer(self, input):
        """Model function for dense CNN"""
        tmp = np.prod(input.shape.as_list()[1:])
        flattened = tf.reshape(input, [-1, tmp])
        ##########################TO DO #####################
        dense_out = tf.layers.dense(flattened,
                                    int(64 * 64 * 64 / 100),
                                    activation=tf.nn.sigmoid)
        return dense_out

    def buildModel(self):
        self.training_data = self.dh.getTrainingData()
        self.input_shape = self.dh.getInputShape()
        self.output_shape = int(np.prod(self.dh.getOutputShape())/100)


        self.input_layer = tf.placeholder(tf.float32,[None,*self.input_shape,1])
        self.output = tf.placeholder(tf.float32,[None,self.output_shape])

        self.conv1 = tf.nn.conv2d(self.input_layer,
                                  tf.random_normal([3,3,1,10]),
                                  strides=[1,2,2,1],
                                  padding='SAME')


        tmp = np.prod(self.conv1.shape.as_list()[1:])
        self.flat = tf.reshape(self.conv1, [-1,tmp])

        self.dense_out = tf.layers.dense(self.flat,
                                     self.output_shape, 
                                    activation=tf.nn.sigmoid)

        self.loss = tf.reduce_mean(tf.square(tf.subtract(self.dense_out, self.output)))

        self.train_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.loss)


    def training(self, iteration = 10):

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())

        for e in range(100):
            self.batch = self.dh.get_batch()
            inp = np.reshape(self.batch[0][0],[2,54,96,1])
            l = self.sess.run([self.loss],
                          feed_dict= {self.input_layer:inp,
                                      self.output:self.batch[1]})
            print(l)

