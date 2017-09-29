from data_handler import Data_handler as DH
from model import Model
import numpy as np
import binvox_rw
import scipy.ndimage

in_path = '/media/nilus/INTENSO/ANN/shit/**/'
voxelDescale = 3
batch_size = 5
img_scaling = 0.1
dh = DH(in_path = in_path, imgScaling = img_scaling, voxelDescaling = voxelDescale, batch_size = batch_size)

M = Model(dataHandler=dh)

print(dh.getInputShape())
batch,_ = dh.get_batch()
print(batch[0].shape)
print(int(np.prod(dh.getOutputShape())))


print("start training")

M.buildModel()
M.training(200)
print("training done")
M.predict('training')

final_output = M.final_output[0]
final_output = np.reshape(final_output, (batch_size, *dh.getOutputShape()))
final_originals = M.batch[1]
np.save('outputs', final_output)
np.save('originals', final_originals)
print("done")

