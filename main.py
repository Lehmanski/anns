import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 
from model import Model 

from os import path, listdir

from dataHandler import DataHandler as DH 




in_path = '/home/me/Datasets/mixed_dataset_subset/'

dh = DH(in_path = in_path, scaling = 0.05, batch_size = 20)
dh.read()
X,Y = dh.next()
print(X[0].shape)
print(Y[0].shape)

'''
# tensorflow part
'''

M = Model(input_shape=X[0].shape, output_shape=Y[0].shape, dataHandler=dh)


for e in [1,20,80,1000]:
	M.training(epochs=e)
	M.predict()

	for ix,o in enumerate(M.output[0]):
		if ix > 4:
			continue
		o = o.reshape(Y[0].shape)
		p = M.targets[ix].reshape(Y[0].shape)
		t = np.hstack((o,p))
		plt.imshow(t)
		plt.colorbar()
		plt.show()