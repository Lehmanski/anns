import numpy as np
import tensorflow as tf
import glob
import os.path
import binvox_rw
import random as ra

batch_size = 1
data_split = 0.8
iterations = 1

def single_cnn(input):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(input, [-1, 540, 960, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=10,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[4, 4], strides=2)
    # Droput Layer #1
    dropout1 = tf.layers.dropout(
        inputs=pool1,
        rate=0.4,
        noise_shape=None,
        seed=None,
        training=False,
        name=None
    )
    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(
        inputs=dropout1,
        filters=10,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)
    # Pooling Layer #2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    return pool2


def combine_layer(inp0, inp1, inp2, inp3, inp4, inp5, inp6, inp7):
    ''' combine the 8 layers to one single concatinated layer'''
    comb_layer = tf.concat([inp0,inp1,inp2,inp3, inp4, inp5, inp6, inp7], 0)
    return comb_layer


def comb_conv(input):
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input,
        filters=10,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[4, 4], strides=2)
    # Droput Layer #1
    dropout1 = tf.layers.dropout(
        inputs=pool1,
        rate=0.4,
        noise_shape=None,
        seed=None,
        training=False,
        name=None
    )
    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(
        inputs=dropout1,
        filters=10,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    # Pooling Layer #2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    # Convolutional Layer #3
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=16,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)
    # Droput Layer #2
    dropout2 = tf.layers.dropout(
        inputs=conv3,
        rate=0.4,
        noise_shape=None,
        seed=None,
        training=False,
        name=None
    )
    tmp = np.prod(dropout2.shape.as_list()[1:])
    flattened = tf.reshape(dropout2, [-1, tmp])
    # Dense Layer # 1
    logits = tf.layers.dense(inputs=flattened, units=262144, activation=tf.nn.relu)
    return logits

def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def get_data():
    """ returns a list of filedirectories where the data is stored """
    fileDirs = []
    ra.seed(2)
    for filename in glob.iglob('/media/nilus/INTENSO/ANN/shit/**/*.npz', recursive=True):
        dir = os.path.dirname(filename)
        fileDirs.append(dir)
    ra.shuffle(fileDirs)
    data = fileDirs[0:int(len(fileDirs)*data_split)]
    test_data = fileDirs[int(len(fileDirs)*data_split):len(fileDirs)]
    return data, test_data


def get_batch(data, size):
    """ returns batches in form of np arrays of shape (batch_size 8*540*960, btach_size 64*64*64) """
    batch_npz = []
    batch_target = []
    for i in range(size):
        random = ra.randint(0, len(data)-1)
        fileDir = data[random]
        npz = np.load(fileDir + '/depth_maps.npz')
        depth_maps = []
        for l in range(len(npz.files)):
            # append the flattened depth image
            depth_maps.append((npz['arr_' + str(l)]).flatten())
        batch_npz.append(np.asarray(depth_maps))
        with open(fileDir+ '/model.binvox', 'rb') as f:
            model = binvox_rw.read_as_3d_array(f)
            batch_target.append(model.data.flatten())
    return np.asarray(batch_npz), np.asarray(batch_target)

if __name__ == "__main__":
    data, test_data = get_data()
    x_0 = tf.placeholder(tf.float32, [None, 518400])
    x_1 = tf.placeholder(tf.float32, [None, 518400])
    x_2 = tf.placeholder(tf.float32, [None, 518400])
    x_3 = tf.placeholder(tf.float32, [None, 518400])
    x_4 = tf.placeholder(tf.float32, [None, 518400])
    x_5 = tf.placeholder(tf.float32, [None, 518400])
    x_6 = tf.placeholder(tf.float32, [None, 518400])
    x_7 = tf.placeholder(tf.float32, [None, 518400])
    # TO DO CHANGE
    # Define loss and optimizer
    # 64 * 64 * 64
    y_ = tf.placeholder(tf.float32, [None, 262144])
    # the CNN for each image invidually
    conv_0 = single_cnn(x_0)
    conv_1 = single_cnn(x_1)
    conv_2 = single_cnn(x_2)
    conv_3 = single_cnn(x_3)
    conv_4 = single_cnn(x_4)
    conv_5 = single_cnn(x_5)
    conv_6 = single_cnn(x_6)
    conv_7 = single_cnn(x_7)
    print(conv_0.get_shape())
    # the concatenation of each layer
    conc_layer = combine_layer(conv_0, conv_1, conv_2, conv_3, conv_4, conv_5, conv_6, conv_7)
    print(conc_layer.get_shape())
    logits = comb_conv(conc_layer)

    ##################
    with tf.name_scope('loss'):
        cross_entropy = loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=logits)
        cross_entropy = tf.reduce_mean(cross_entropy)
        print(cross_entropy)

        with tf.name_scope('adam_optimizer'):
            train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    ##################

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(iterations):
            batch = get_batch(data, batch_size)
            train_step.run(feed_dict={
                x_0: batch[0][0],
                x_1: batch[0][1],
                x_2: batch[0][2],
                x_3: batch[0][3],
                x_4: batch[0][4],
                x_5: batch[0][5],
                x_6: batch[0][6],
                x_7: batch[0][7],
                y_: batch[1]})

    batch = get_batch(data, batch_size)
    print()


