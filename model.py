import numpy as np
import tensorflow as tf

from dataHandler import DataHandler as DH 

class Model():
	def __init__(self, input_shape, output_shape,dataHandler=None):
		self.input_shape = input_shape
		self.output_shape = output_shape
		self.dh = dataHandler
		# graph parameters
		self.size_patch_1 = 7
		self.size_patch_2 = 5
		self.size_patch_3 = 5
		self.size_patch_4 = 3
		self.size_patch_5 = 3

		self.n_out_channels_1 = 8 # output channels frist layer
		self.n_out_channels_2 = 16 # output channels second layer
		self.n_out_channels_3 = 16 # output channels third layer
		self.n_out_channels_4 = 32 # output channels fourth layer
		self.n_out_channels_5 = 64 # output channels firth layer

		self.n_dense_neurons_1 = 512
		self.n_dense_neurons_2 = 2048
		self.n_dense_neurons_out = np.multiply(*self.output_shape)# number of densely connected neurons in output layer
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
								 tf.random_uniform([self.size_patch_1,self.size_patch_1,3,self.n_out_channels_1]),
								 strides=[1,2,2,1],
								 padding='SAME')

		self.conv1_pool = tf.nn.max_pool(self.conv1,
										 ksize=[1,2,2,1],
										 strides=[1,2,2,1],
										 padding='SAME')

		self.conv1_drop = tf.nn.dropout(self.conv1_pool, 0.6)


		'''
		Second Layer
		'''
		self.conv2 = tf.nn.conv2d(self.conv1_drop,
								 tf.random_uniform([self.size_patch_2,self.size_patch_2,self.n_out_channels_1,self.n_out_channels_2]),
								 strides=[1,2,2,1],
								 padding='SAME')

		self.conv2_pool = tf.nn.max_pool(self.conv2,
										 ksize=[1,2,2,1],
										 strides=[1,2,2,1],
										 padding='SAME')

		self.conv2_drop = tf.nn.dropout(self.conv2_pool, 0.6)


		'''
		Third Layer
		'''
		#self.conv3 = tf.layers.conv2d(self.conv2_drop,
		#							64,
		#						 [5,5],
		#						 strides=[1,1],
		#						 padding='SAME',
		#						 activation=None)
		self.conv3 = tf.nn.conv2d(self.conv2_drop,
								 tf.random_uniform([self.size_patch_3,self.size_patch_3,self.n_out_channels_2,self.n_out_channels_3]),
								 strides=[1,2,2,1],
								 padding='SAME')

		self.conv3_pool = tf.nn.max_pool(self.conv3,
										 ksize=[1,2,2,1],
										 strides=[1,2,2,1],
										 padding='SAME')

		self.conv3_drop = tf.nn.dropout(self.conv3_pool, 0.6)
		'''
		Fourth Layer
		'''
		self.conv4 = tf.nn.conv2d(self.conv3_drop,
								 tf.random_uniform([self.size_patch_4,self.size_patch_4,self.n_out_channels_3,self.n_out_channels_4]),
								 strides=[1,2,2,1],
								 padding='SAME')

		self.conv4_pool = tf.nn.max_pool(self.conv4,
										 ksize=[1,2,2,1],
										 strides=[1,2,2,1],
										 padding='SAME')

		self.conv4_drop = tf.nn.dropout(self.conv4_pool, 0.6)
		'''
		Fifth Layer
		'''
		self.conv5 = tf.nn.conv2d(self.conv4_drop,
								 tf.random_uniform([self.size_patch_5,self.size_patch_5,self.n_out_channels_4,self.n_out_channels_5]),
								 strides=[1,2,2,1],
								 padding='SAME')

		self.conv5_pool = tf.nn.max_pool(self.conv5,
										 ksize=[1,2,2,1],
										 strides=[1,2,2,1],
										 padding='SAME')

		self.conv5_drop = tf.nn.dropout(self.conv5_pool, 0.6)

		'''
		Fully Connected Output Layer
		'''
		# reshape layer before
		tmp = np.prod(self.conv5_drop.shape.as_list()[1:])
		self.flat = tf.reshape(self.conv5_drop, [-1, tmp])

		self.dense1 = tf.layers.dense(self.flat,
									  self.n_dense_neurons_1,
									  activation=tf.nn.tanh)
		self.dense1_drop = tf.nn.dropout(self.dense1, 0.6)
		'''
		self.dense2 = tf.layers.dense(self.dense1,
									  self.n_dense_neurons_2,
									  activation=tf.nn.relu)
		'''

		self.dense_out = tf.layers.dense(self.dense1_drop, 
									self.n_dense_neurons_out, 
									activation=tf.nn.sigmoid)

		'''
		Loss
		'''
		self.loss = tf.reduce_mean(tf.square(self.dense_out-self.target))

		self.train_step = tf.train.AdamOptimizer(learning_rate=0.1).minimize(self.loss)

		self.sess = tf.Session()
		self.init = tf.global_variables_initializer()
		self.sess.run(self.init)


	def predict(self, type='test'):
		if type  == 'test':
			X,Y = self.dh.testingData()
		elif type == 'train':
			X,Y = self.dh.next(12)

		self.X = np.reshape(X, [-1,*X[0].shape])
		self.Y = np.reshape(Y,[-1,np.multiply(*Y[0].shape)])
		self.feed_dict = {self.input_layer: self.X,
							  self.target: self.Y}

		self.output = self.sess.run([self.dense_out],
						  	feed_dict=self.feed_dict)



	def training(self,epochs=10,batch_size=100):
		hist = []
		mean_over = 10
		for i in range(epochs):
			X,Y = self.dh.next(batch_size=batch_size)
			inp = np.reshape(X, [-1,*X[0].shape])
			target = np.reshape(Y,[-1,np.multiply(*Y[0].shape)])
			
			self.feed_dict = {self.input_layer: inp,
							  self.target: target}

			_,l = self.sess.run([self.train_step,self.loss],
						  feed_dict=self.feed_dict)
			hist.append(l)
			print(i,l)
			if (i+1) % mean_over == 0:
				print('mean over last {0} epochs: {1}'.format(len(hist),np.mean(hist)))
				hist = []


			