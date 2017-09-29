from data_handler import Data_handler as DH
from model import Model
import numpy as np

in_path = '/media/nilus/INTENSO/ANN/shit/**/'

dh = DH(in_path = in_path, scaling = 0.1, batch_size = 2)

M = Model(dataHandler=dh)

print(dh.getInputShape())
print(dh.get_batch()[0].shape)
print(dh.get_batch()[0][0].shape)

print("start training")
M.training(2)
print("done")

