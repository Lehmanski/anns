import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 
from model import Model 
import time

from os import path, listdir

from dataHandler import DataHandler as DH 




in_path = '/home/me/Datasets/depth_linus/mixed_dataset_mini/'
in_path = '/home/me/Datasets/rgbd-scenes/desk/'

dh = DH(in_path = in_path, scaling = 0.5, batch_size = 10)
#dh.read()
dh.read2()
X,Y = dh.next()
print(X[0].shape)
print(Y[0].shape)

'''
# tensorflow part
'''

M = Model(input_shape=X[0].shape, output_shape=Y[0].shape, dataHandler=dh)

l_hist = []
for e in [5,100,100,100,100]:
	M.training(epochs=e)
	M.predict('test')

	for ix,o in enumerate(M.output[0]):
		if ix > 20:
			continue
		o = o.reshape(Y[0].shape)
		p = M.Y[ix].reshape(Y[0].shape)
		q = M.X[ix]
		f, ax = plt.subplots(2,2)
		ax[0,0].imshow(q)
		ax[1,0].imshow(o)
		ax[1,1].imshow(p)
		f.show()
		time.sleep(2.5)
		plt.close(f)
