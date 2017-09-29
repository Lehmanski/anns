from os import path, listdir
import numpy as np
from scipy.misc import imresize, imread



class DataHandler():
	def __init__(self, in_path=None, split=0.75, batch_size=100, scaling=0.1):
		self.in_path = in_path
		self.split = split
		self.batch_size = batch_size
		self.scaling = scaling


	def read2(self, in_path=None):
		data_holder = {}
		self.training_data_holder = []
		self.testing_data_holder = []

		if self.in_path is None and in_path is None:
			print('no input path given. Specify "in_path="')
			return
		else:
			if self.in_path is None and not in_path is None:
				self.in_path = in_path


		self.folders = listdir(self.in_path)
		self.folders = [f for f in self.folders if path.isdir(path.join(self.in_path,f))]
		for folder in self.folders:
			j = path.join(self.in_path,folder)
			files = listdir(j)
			# full paths to image files
			image_paths = sorted([path.join(j,f) for f in files if not f.endswith('depth.png')])
			# the depth maps as dict of arrays
			depth_paths = [o.split('.')[0]+'_depth.png' for o in image_paths]
			depths = [imresize(imread(p),self.scaling) for p in depth_paths]
			# normalize depths (0,1)
			tmp = []
			for d in depths:
				d = d.astype('float32')
				d -= d.min()
				d /= d.max()
				tmp.append(d)
			depths = tmp
			del tmp

			for p,d in zip(image_paths,depths):
				if d.max()==0:
					continue
				if np.any(np.isnan(d)):
					continue
				data_holder[p] = [p,d]

		self.n_items = len(data_holder.keys())

		idx = np.random.permutation(self.n_items)
		split_idx = int(self.n_items*self.split)

		data_holder = list(data_holder.items())
		self.training_data_holder = [data_holder[i][1] for i in idx[:split_idx]]
		self.testing_data_holder = [data_holder[i][1] for i in idx[split_idx:]]

		print('{0} items. {1} for training, {2} for testing'.format(self.n_items, len(idx[:split_idx]), len(idx[split_idx:])))





	def read(self, in_path = None):
		data_holder = {}
		self.training_data_holder = []
		self.testing_data_holder = []
		'''
		To save memory, we will only load paths to image_paths and load an image
		on demand while training.
		Depth maps will be loaded immediately 
		'''
		if self.in_path is None and in_path is None:
			print('no input path given. Specify "in_path="')
			return
		else:
			if self.in_path is None and not in_path is None:
				self.in_path = in_path
		# list folders in main folder
		self.folders = listdir(self.in_path)
		self.folders = [f for f in self.folders if path.isdir(path.join(self.in_path,f))]
		for folder in self.folders:
			j = path.join(self.in_path,folder)
			files = listdir(j)
			# full paths to image files
			image_paths = sorted([path.join(j,f) for f in files if f.endswith('.png')])
			# the depth maps as dict of arrays
			d_map_path = path.join(j,[f for f in files if f.endswith('npz')][0])
			d_map = np.load(d_map_path)
			keys = d_map.keys()
			data = d_map.items()
			# argsort, so that they line up with image_paths
			idx = np.argsort(keys)
			keys = [keys[i] for i in idx]
			data = [data[i][1] for i in idx]
			for p,d in zip(image_paths,data):
				d = imresize(d, self.scaling)/255
				if d.max()==0:
					continue
				if d.mean()<0.1:
					continue
				if d.mean()>0.9:
					continue
				data_holder[p] = [p,d]

		self.n_items = len(data_holder.keys())

		idx = np.random.permutation(self.n_items)
		split_idx = int(self.n_items*self.split)

		data_holder = list(data_holder.items())
		self.training_data_holder = [data_holder[i][1] for i in idx[:split_idx]]
		self.testing_data_holder = [data_holder[i][1] for i in idx[split_idx:]]

		print('{0} items. {1} for training, {2} for testing'.format(self.n_items, len(idx[:split_idx]), len(idx[split_idx:])))



	def next(self, batch_size=None):
		X = []
		Y = []
		if not batch_size is None:
			idx = np.random.randint(len(self.training_data_holder),size=[batch_size])
		else:
			idx = np.random.randint(len(self.training_data_holder),size=[self.batch_size])

		for i in idx:
			d = self.training_data_holder[i]
			image = imresize(imread(d[0]),self.scaling)
			if image.shape[-1] == 4:
				image = image[:,:,:-1]
			depth_map = d[1]
			X.append(image)
			Y.append(depth_map)
		X = np.array(X)
		Y = np.array(Y)
		return X,Y

	def testingData(self):
		X = []
		Y = []
		for i in range(len(self.testing_data_holder)):
			d = self.training_data_holder[i]
			image = imresize(imread(d[0]),self.scaling)
			if image.shape[-1] == 4:
				image = image[:,:,:-1]
			depth_map = d[1]
			X.append(image)
			Y.append(depth_map)
		X = np.array(X)
		Y = np.array(Y)
		return X,Y

