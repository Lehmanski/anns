import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 
from model import Model 

from os import path, listdir

from dataHandler import DataHandler as DH 




in_path = '/home/me/Datasets/mixed_dataset/'

dh = DH(in_path = in_path, scaling = 0.1, batch_size = 20)
dh.read()
X,Y = dh.next()
print(X[0].shape)
print(Y[0].shape)

'''
# tensorflow part
'''

M = Model(input_shape=X[0].shape, output_shape=Y[0].shape, dataHandler=dh)


for e in [100,200,1000]:
	M.training(epochs=e)
	M.predict()

	for o in M.output[0]:
		o = o.reshape(Y[0].shape)
		plt.imshow(o)
		plt.show()