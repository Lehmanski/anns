import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 
from model import Model 
import time

from os import path, listdir, makedirs

from dataHandler import DataHandler as DH 

import sys

#'/home/me/Datasets/depth_linus/mixed_dataset_subset/'

class main():

	def __init__(self,in_path,fig_path):
		r_key = np.random.randint(3000)
		#in_path = '/home/me/Datasets/rgbd-scenes/background/'
		bs=5
		dh = DH(in_path = in_path, scaling = 0.2, batch_size = bs)
		dh.read()
		#dh.read2()
		X,Y = dh.next(1)
		print(X[0].shape)
		print(Y[0].shape)

		'''
		# tensorflow part
		'''
		self.M = Model(input_shape=X[0].shape, output_shape=Y[0].shape, dataHandler=dh)
		return 
		for r in range(200):
			for e in [bs]:
				self.M.training(epochs=e)
			self.M.predict('test')
			#
			tmp = []
			for o in self.M.output[0]:
				tmp.append(o.reshape(self.M.output_shape))

			o = np.hstack(tmp)
			p = np.hstack(self.M.Y.reshape(-1,*self.M.output_shape))
			q = np.hstack(self.M.X)
			#
			f, ax = plt.subplots(1,1)
			#
			#ax[0].imshow(o)
			#ax[1].imshow(p)
			#ax[2].imshow(q)
			#
			ax.imshow(np.vstack((o,p)))
			mng = plt.get_current_fig_manager()
			mng.resize(*mng.window.maxsize())
			if not path.isdir(fig_path):
				makedirs(fig_path)
			f.savefig(path.join(fig_path,'{1}_repetition_{0}.png'.format(r,r_key)),dpi=200)
				#f.show()
				#time.sleep((e/5))
				#plt.close(f)



if __name__ == '__main__':
	m = main(sys.argv[1],sys.argv[2])