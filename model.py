import numpy as np
import tensorflow as tf

from dataHandler import DataHandler as DH 

class Model():
	def __init__(self, input_shape, output_shape):
		self.input_shape = input_shape
		self.output_shape = output_shape

		# graph parameters
		self.size_patch_1 = 5
		self.size_patch_2 = 5
		self.size_patch_3 = 5
		self.n_out_channels_1 = 32 # output channels frist layer
		self.n_out_channels_2 = 64 # output channels second layer
		self.n_out_channels_3 = 32 # output channels second layer
		self.n_dense_neurons = np.multiply(*self.output_shape)# number of densely connected neurons in output layer
		self.n_outputs = np.multiply(*self.output_shape)# 



		'''
		Input
		'''
		# input variable
		self.input_layer = tf.placeholder(tf.float32,[None, *input_shape])

		self.target = tf.placeholder(tf.float32, [None, np.multiply(*output_shape)])



		'''
		First Layer
		'''
		self.conv1 = tf.nn.conv2d(self.input_layer,
								 tf.random_normal([self.size_patch_1,self.size_patch_1,3,self.n_out_channels_1]),
								 strides=[1,2,2,1],
								 padding='SAME')
		'''
		Second Layer
		'''
		self.conv2 = tf.nn.conv2d(self.conv1,
								 tf.random_normal([self.size_patch_2,self.size_patch_2,self.n_out_channels_1,self.n_out_channels_2]),
								 strides=[1,2,2,1],
								 padding='SAME')

		'''
		Third Layer
		'''
		self.conv3 = tf.nn.conv2d(self.conv2,
								 tf.random_normal([self.size_patch_3,self.size_patch_3,self.n_out_channels_2,self.n_out_channels_3]),
								 strides=[1,2,2,1],
								 padding='SAME')
		print(self.conv3.shape)

		'''
		Fully Connected Output Layer
		'''
		# reshape layer before
		self.conv3_flat = tf.reshape(self.conv3, [-1, 17*30*32])
		print(self.conv3_flat.shape)

		self.dense = tf.layers.dense(self.conv3_flat, 
									self.n_dense_neurons, 
									activation=tf.nn.relu)

		'''
		Dropout Layer
		'''
		"""
		self.dropout = tf.layers.dropout(self.dense, 
										rate=0.4)
		"""

		'''
		Loss
		'''
		self.loss = tf.reduce_mean(tf.square(self.dense-self.target))

		self.train_step = tf.train.GradientDescentOptimizer(learning_rate=0.003).minimize(self.loss)

		

	def training(self,dh):
		self.sess = tf.Session()
		self.init = tf.global_variables_initializer()
		self.sess.run(self.init)
		for i in range(100):
			X,Y = dh.next()
			inp = np.reshape(X, [-1,135,240,3])
			target = np.reshape(Y,[-1,135*240])
			
			self.feed_dict = {self.input_layer: inp,
							  self.target: target}

			_,l = self.sess.run([self.train_step,self.loss],
						  feed_dict=self.feed_dict)
			print(i,l)


			