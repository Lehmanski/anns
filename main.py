import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 
from model import Model 
import time

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

l_hist = []
for e in [100,100,100,100,100]:
	M.training(epochs=e)
	M.predict('train')

	for ix,o in enumerate(M.output[0]):
		if ix > 4:
			continue
		o = o.reshape(Y[0].shape)
		p = M.Y[ix].reshape(Y[0].shape)
		q = M.X[ix]
		f, (ax1, ax2, ax3) = plt.subplots(1,3)
		ax1.imshow(o)
		ax2.imshow(q)
		ax3.imshow(p)
		f.show()
		time.sleep(5)
		plt.close(f)
