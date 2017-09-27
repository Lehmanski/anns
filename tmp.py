import glob	
import os 

counter = 0
for filename in glob.iglob('/home/me/Datasets/mixed_dataset/**/*.npz', recursive=True):
        dir = os.path.dirname(filename)
        counter += 1
print(counter)