from data_handler import Data_handler as DH
from model import Model
import numpy as np

in_path = '/media/nilus/INTENSO/ANN/shit/**/'

dh = DH(in_path = in_path, imgScaling = 0.1, voxelDescaling = 2, batch_size = 10)

M = Model(dataHandler=dh)

print(dh.getInputShape())
batch,_ = dh.get_batch()
print(batch[0].shape)
print(int(np.prod(dh.getOutputShape())))


print("start training")

M.buildModel()
M.training(100)

print("done")

