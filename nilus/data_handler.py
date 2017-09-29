import random as ra
import glob
import os.path
import numpy as np
import scipy.misc
import binvox_rw
import scipy.ndimage

class Data_handler():
    def __init__(self, in_path=None, split=0.7, imgScaling=0.1, voxelDescaling = 4, batch_size=10):
        """ returns a list of filedirectories where the data is stored """
        fileDirs = []
        #ra.seed(2)
        for filename in glob.iglob(in_path+'*.npz', recursive=True):
            dir = os.path.dirname(filename)
            fileDirs.append(dir)
        ra.shuffle(fileDirs)
        self.data = fileDirs[0:int(len(fileDirs)*split)]
        self.test_data = fileDirs[int(len(fileDirs)*split):len(fileDirs)]
        self.scaling = imgScaling
        self.batch_size = batch_size
        self.voxelDescale = voxelDescaling

    def getTrainingData(self):
        return self.data

    def getTestData(self):
        return self.test_data

    def get_batch(self):
        """ returns batches in form of np arrays of shape (batch_size 8*540*960, btach_size 64*64*64) """
        batch_target = []
        depth_maps = [[] for i in range(8)] # hardcoded for now
        for i in range(self.batch_size):
            random = ra.randint(0, len(self.data)-1)
            fileDir = self.data[random]
            npz = np.load(fileDir + '/depth_maps.npz')
            for l in range(len(npz.files)):
                # append the flattened depth image
                image = npz['arr_' + str(l)]
                image = scipy.misc.imresize(image, self.scaling, interp='bilinear', mode=None)
                depth_maps[l].append(np.asarray(image))
            with open(fileDir+ '/model.binvox', 'rb') as f:
                model = binvox_rw.read_as_3d_array(f)
                outarr = scipy.ndimage.zoom(model.data, (1/self.voxelDescale))
                arr = outarr.flatten().astype(float)
                batch_target.append(arr)
        return np.asarray(depth_maps), np.asarray(batch_target)

    def getInputShape(self):
        fileDir = self.data[0]
        npz = np.load(fileDir + '/depth_maps.npz')
        image = npz['arr_' + str(0)]
        image = scipy.misc.imresize(image, self.scaling, interp='bilinear', mode=None)
        return np.asarray(image).shape

    def getOutputShape(self):
        fileDir = self.data[0]
        with open(fileDir + '/model.binvox', 'rb') as f:
            model = binvox_rw.read_as_3d_array(f)
            arr = model.data
        outarr = scipy.ndimage.zoom(model.data, (1 / self.voxelDescale))
        return outarr.shape


