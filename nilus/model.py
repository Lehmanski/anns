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


    def training(self, iteration = 10):
        training_data = self.dh.getTrainingData()

        x_0 = tf.placeholder(tf.float32, [None, *self.dh.getInputShape()])
        y_ = tf.placeholder(tf.float32, [None, len(training_data[1][0])])

        conv1 = self.cnn_forDepthMaps(x_0)
        dense = self.dense_layer(conv1)

        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.square(dense- y_))
            with tf.name_scope('adam_optimizer'):
                train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

        with tf.Session() as sess:
            print("sess started")
            sess.run(tf.global_variables_initializer())
            for i in range(iteration):
                batch = self.dh.get_batch()
                input = np.reshape(batch[0], [-1, *batch[0][0].shape, 1])
                _, lossl = train_step.run([train_step, loss], feed_dict={
                    x_0: input,
                    y_: batch[1]})
                print(lossl)
