import numpy as np
import tensorflow as tf
from model import Model 

from os import path, listdir

from dataHandler import DataHandler as DH 




in_path = '/home/me/Datasets/mixed_dataset_subset/'

dh = DH(in_path = in_path, scaling = 0.25, batch_size = 20)
dh.read()
X,Y = dh.next()
print(X[0].shape)
print(Y[0].shape)

'''
# tensorflow part
'''

M = Model(input_shape=X[0].shape, output_shape=Y[0].shape)


for i in range(100):
	M.training(dh)
