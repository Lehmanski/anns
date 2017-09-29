import numpy as np
import tensorflow as tf


class InputLayer():

	def __init__(self,
				input_shape,
				kernel_size = [5,5],
				out_channels = 16,
				strides = [1,2,2,1],
				padding = 'SAME'):

		self.kernel_size = kernel_size
		self.out_channels = out_channels
		self.strides = strides
		self.padding = padding


		self.input = tf.placeholder(tf.float32,
									[None, *input_shape,1])

		self.conv1 = tf.nn.conv2d(self.input,
								tf.random_normal([*self.kernel_size,1,self.out_channels]),
								strides=self.strides,
								padding=self.padding
								)

		self.conv1_pool = tf.nn.max_pool(self.conv1,
										 ksize=[1,2,2,1],
										 strides=[1,2,2,1],
										 padding='SAME')

		self.conv1_drop = tf.nn.dropout(self.conv1_pool, 0.6)


class Model():

	def __init__(self, dh):

		self.dh = dh

		x,y = self.dh.get_batch()

		self.input_shape = x[0][0].shape
		self.output_shape = y[0].shape[0]


		self.target = tf.placeholder(tf.float32, [None, self.output_shape])

		'''
		Input Layers

		Create a list of input layers.
		Each input layer is a convolutional layer with maxpool and dropout
		'''
		self.input_layers = []
		#
		self.i0 = InputLayer(self.input_shape)
		self.i1 = InputLayer(self.input_shape)
		self.i2 = InputLayer(self.input_shape)
		self.i3 = InputLayer(self.input_shape)
		self.i4 = InputLayer(self.input_shape)
		self.i5 = InputLayer(self.input_shape)
		self.i6 = InputLayer(self.input_shape)
		self.i7 = InputLayer(self.input_shape)

		# concatenate into one layer
		self.conc_layer = tf.concat([self.i0.conv1_drop,
									 self.i1.conv1_drop,
									 self.i2.conv1_drop,
									 self.i3.conv1_drop,
									 self.i4.conv1_drop,
									 self.i5.conv1_drop,
									 self.i6.conv1_drop,
									 self.i7.conv1_drop], 1)
		
		'''
		Convolutional Layer
		'''

		self.conv1 = tf.nn.conv2d(self.conc_layer,
								tf.random_normal([3,3,int(self.conc_layer.shape[-1]),16]),
								strides=[1,2,2,1],
								padding='SAME')

		self.conv1_pool = tf.nn.max_pool(self.conv1,
										 ksize=[1,2,2,1],
										 strides=[1,2,2,1],
										 padding='SAME')

		self.conv1_drop = tf.nn.dropout(self.conv1_pool, 0.6)


		'''
		Flattening
		'''
		tmp = np.prod(self.conv1_drop.shape.as_list()[1:])
		self.flat = tf.reshape(self.conv1_drop,[-1, tmp])

		'''
		Dense Layer
		'''
		self.dense_out = tf.layers.dense(self.flat, 
									self.output_shape, 
									activation=tf.nn.sigmoid)



		'''
		Loss
		'''
		self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.target,
																		  logits=self.dense_out,
																		  ))

		self.train_step = tf.train.AdamOptimizer(learning_rate=0.005).minimize(self.loss)

		'''
		Initialization
		'''
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

	def train(self,epochs=100):
		l_hist = []
		for e in range(epochs):
			X,Y = self.dh.get_batch()

			feed_dict = {self.i0.input:np.reshape(X[0,:,:,:],[-1,*self.input_shape,1]),
						 self.i1.input:np.reshape(X[1,:,:,:],[-1,*self.input_shape,1]),
						 self.i2.input:np.reshape(X[2,:,:,:],[-1,*self.input_shape,1]),
						 self.i3.input:np.reshape(X[3,:,:,:],[-1,*self.input_shape,1]),
						 self.i4.input:np.reshape(X[4,:,:,:],[-1,*self.input_shape,1]),
						 self.i5.input:np.reshape(X[5,:,:,:],[-1,*self.input_shape,1]),
						 self.i6.input:np.reshape(X[6,:,:,:],[-1,*self.input_shape,1]),
						 self.i7.input:np.reshape(X[7,:,:,:],[-1,*self.input_shape,1]),
						 self.target:Y}

			_,l=self.sess.run([self.train_step,self.loss],
					 feed_dict=feed_dict)
			l_hist.append(l)
			if e%10==0:
				print('mean over 10: {0}'.format(np.mean(l_hist)))
				l_hist = []
			print(np.sum(l))


