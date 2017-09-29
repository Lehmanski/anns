from data_handler import Data_handler as DH
from newModel import Model
import numpy as np

in_path = '/home/me/Datasets/depth_linus/mixed_dataset_subset/**/'

dh = DH(in_path = in_path, imgScaling = 0.1, voxelDescaling = 2, batch_size = 2)

M = Model(dh)

print(dh.getInputShape())
batch,_ = dh.get_batch()
print(batch[0].shape)
print(int(np.prod(dh.getOutputShape())))


print("start training")
M.train(epochs = 1000)

print("done")

