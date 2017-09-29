import numpy as np
import tensorflow as tf

class Model():
    def __init__(self, dataHandler=None):
        self.dh = dataHandler


    def cnn_forDepthMaps(self, input):
        """Model function for CNN."""
        self.convo1 = tf.nn.conv2d(input,
                     tf.random_normal([5, 5, 1, 10]),
                     strides=[1, 2, 2, 1],
                     padding='SAME')
        self.convo1_pool = tf.nn.max_pool(self.convo1,
                                         ksize=[1, 2, 2, 1],
                                         strides=[1, 2, 2, 1],
                                         padding='SAME')
        self.convo1_drop = tf.nn.dropout(self.convo1_pool, 0.6)
        self.convo2 = tf.layers.conv2d(inputs=self.convo1_drop,
                                        filters=10,
                                        kernel_size=[3, 3],
                                        padding="same",
                                        activation=tf.nn.relu)
        self.convo2_pool = tf.nn.max_pool(self.convo2,
                                         ksize=[1, 2, 2, 1],
                                         strides=[1, 2, 2, 1],
                                         padding='SAME')
        self.convo2_drop = tf.nn.dropout(self.convo2_pool, 0.6)
        return self.convo2_drop


    def dense_layer(self, input):
        """Model function for dense CNN"""
        self.convo1 = tf.layers.conv2d(inputs=input,
                                      filters=10,
                                      kernel_size=[5, 5],
                                      padding="same",
                                      activation=tf.nn.relu)
        self.convo1_pool = tf.nn.max_pool(self.convo1,
                                         ksize=[1, 2, 2, 1],
                                         strides=[1, 2, 2, 1],
                                         padding='SAME')
        self.convo1_drop = tf.nn.dropout(self.convo1_pool, 0.8)
        self.convo2 = tf.layers.conv2d(inputs=self.convo1_drop,
                                       filters=32,
                                       kernel_size=[3, 3],
                                       padding="same",
                                       activation=tf.nn.relu)
        self.convo2_pool = tf.nn.max_pool(self.convo2,
                                          ksize=[1, 2, 2, 1],
                                          strides=[1, 2, 2, 1],
                                          padding='SAME')
        self.convo2_drop = tf.nn.dropout(self.convo2_pool, 0.8)
        tmp = np.prod(self.convo2_drop.shape.as_list()[1:])
        self.flat = tf.reshape(self.convo2_drop, [-1, tmp])

        self.dense = tf.layers.dense(self.flat,
                                    np.prod(self.dh.getOutputShape()),
                                    activation=tf.nn.sigmoid)
        return self.dense

    def buildModel(self):
        ''' prepare input and output '''
        self.training_data = self.dh.getTrainingData()
        self.input_shape = self.dh.getInputShape()
        self.output_shape = int(np.prod(self.dh.getOutputShape()))

        self.input_layer_0 = tf.placeholder(tf.float32, [None, *self.input_shape, 1])
        self.input_layer_1 = tf.placeholder(tf.float32, [None, *self.input_shape, 1])
        self.input_layer_2 = tf.placeholder(tf.float32, [None, *self.input_shape, 1])
        self.input_layer_3 = tf.placeholder(tf.float32, [None, *self.input_shape, 1])
        self.input_layer_4 = tf.placeholder(tf.float32, [None, *self.input_shape, 1])
        self.input_layer_5 = tf.placeholder(tf.float32, [None, *self.input_shape, 1])
        self.input_layer_6 = tf.placeholder(tf.float32, [None, *self.input_shape, 1])
        self.input_layer_7 = tf.placeholder(tf.float32, [None, *self.input_shape, 1])

        self.output = tf.placeholder(tf.float32,[None,self.output_shape])
        ''' network procedure '''
        self.conv0 = self.cnn_forDepthMaps(self.input_layer_0)
        self.conv1 = self.cnn_forDepthMaps(self.input_layer_1)
        self.conv2 = self.cnn_forDepthMaps(self.input_layer_2)
        self.conv3 = self.cnn_forDepthMaps(self.input_layer_3)
        self.conv4 = self.cnn_forDepthMaps(self.input_layer_4)
        self.conv5 = self.cnn_forDepthMaps(self.input_layer_5)
        self.conv6 = self.cnn_forDepthMaps(self.input_layer_6)
        self.conv7 = self.cnn_forDepthMaps(self.input_layer_7)

        self.conc_layer = tf.concat([self.conv0, self.conv1, self.conv2, self.conv3, self.conv4,
                                self.conv5, self.conv6, self.conv7], 0)

        self.dense_out = self.dense_layer(self.conv0)

        with tf.name_scope('loss'):

            #self.loss = tf.reduce_mean(tf.square(tf.subtract(self.dense_out, self.output)))
            self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.output, logits=self.dense_out)
            self.loss = tf.reduce_mean(self.cross_entropy)
            self.train_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.loss)


    def training(self, iteration = 10):

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())

        for e in range(iteration):
            self.batch = self.dh.get_batch()
            batch_size = len(self.batch[0][0])
            inp_0 = np.reshape(self.batch[0][0], [batch_size, *self.input_shape, 1])
            inp_1 = np.reshape(self.batch[0][1], [batch_size, *self.input_shape, 1])
            inp_2 = np.reshape(self.batch[0][2], [batch_size, *self.input_shape, 1])
            inp_3 = np.reshape(self.batch[0][3], [batch_size, *self.input_shape, 1])
            inp_4 = np.reshape(self.batch[0][4], [batch_size, *self.input_shape, 1])
            inp_5 = np.reshape(self.batch[0][5], [batch_size, *self.input_shape, 1])
            inp_6 = np.reshape(self.batch[0][6], [batch_size, *self.input_shape, 1])
            inp_7 = np.reshape(self.batch[0][7], [batch_size, *self.input_shape, 1])
            l = self.sess.run([self.loss],
                          feed_dict= {self.input_layer_0: inp_0,
                                      self.input_layer_1: inp_1,
                                      self.input_layer_2: inp_2,
                                      self.input_layer_3: inp_3,
                                      self.input_layer_4: inp_4,
                                      self.input_layer_5: inp_5,
                                      self.input_layer_6: inp_6,
                                      self.input_layer_7: inp_7,
                                      self.output:self.batch[1]})
            if iteration%10 == 0:
                print(l)

        def predict(self, type='test'):
            if type == 'test':
                print("TO DO")
            elif type == 'train':
                X, Y = self.dh.get_batch

            self.batch = self.dh.get_batch()
            batch_size = len(self.batch[0][0])
            inp_0 = np.reshape(self.batch[0][0], [batch_size, *self.input_shape, 1])
            inp_1 = np.reshape(self.batch[0][1], [batch_size, *self.input_shape, 1])
            inp_2 = np.reshape(self.batch[0][2], [batch_size, *self.input_shape, 1])
            inp_3 = np.reshape(self.batch[0][3], [batch_size, *self.input_shape, 1])
            inp_4 = np.reshape(self.batch[0][4], [batch_size, *self.input_shape, 1])
            inp_5 = np.reshape(self.batch[0][5], [batch_size, *self.input_shape, 1])
            inp_6 = np.reshape(self.batch[0][6], [batch_size, *self.input_shape, 1])
            inp_7 = np.reshape(self.batch[0][7], [batch_size, *self.input_shape, 1])
            self.batch = self.dh.get_batch()
            batch_size = len(self.batch[0][0])
            inp_0 = np.reshape(self.batch[0][0], [batch_size, *self.input_shape, 1])
            inp_1 = np.reshape(self.batch[0][1], [batch_size, *self.input_shape, 1])
            inp_2 = np.reshape(self.batch[0][2], [batch_size, *self.input_shape, 1])
            inp_3 = np.reshape(self.batch[0][3], [batch_size, *self.input_shape, 1])
            inp_4 = np.reshape(self.batch[0][4], [batch_size, *self.input_shape, 1])
            inp_5 = np.reshape(self.batch[0][5], [batch_size, *self.input_shape, 1])
            inp_6 = np.reshape(self.batch[0][6], [batch_size, *self.input_shape, 1])
            inp_7 = np.reshape(self.batch[0][7], [batch_size, *self.input_shape, 1])
            l = self.sess.run([self.loss],
                              feed_dict={self.input_layer_0: inp_0,
                                         self.input_layer_1: inp_1,
                                         self.input_layer_2: inp_2,
                                         self.input_layer_3: inp_3,
                                         self.input_layer_4: inp_4,
                                         self.input_layer_5: inp_5,
                                         self.input_layer_6: inp_6,
                                         self.input_layer_7: inp_7,
                                         self.output: self.batch[1]})
