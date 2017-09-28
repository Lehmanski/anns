import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 
from model import Model 
import time

from os import path, listdir

from dataHandler import DataHandler as DH 




in_path = '/home/me/Datasets/depth_linus/mixed_dataset_mini/'
#in_path = '/home/me/Datasets/rgbd-scenes/desk/'
bs=10
dh = DH(in_path = in_path, scaling = 0.75, batch_size = bs)
dh.read()
#dh.read2()
X,Y = dh.next(1)
print(X[0].shape)
print(Y[0].shape)

'''
# tensorflow part
'''

M = Model(input_shape=X[0].shape, output_shape=Y[0].shape, dataHandler=dh)




for r in range(100):
	for e in [50]:
		M.training(epochs=e)
		M.predict('train')
		#
		o = np.hstack(M.output[0].reshape((*Y[0].T.shape),-1))
		p = np.hstack(M.Y.reshape(bs,*Y[0].shape))
		q = np.hstack(M.X)
		#
		f, ax = plt.subplots(3,1)
		#
		ax[0].imshow(o)
		ax[1].imshow(p)
		ax[2].imshow(q)
		#
		f.show()
		time.sleep(e/5.)
		plt.close(f)