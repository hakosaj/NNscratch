import numpy as np
import struct
import random
from dnn import DeepNeuralNetwork
import time

# https://mlfromscratch.com/neural-network-tutorial/#/
trainsize = 0.8
testsize = 1 - trainsize

with open("images", "rb") as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    imgs = np.fromfile(f, dtype=np.dtype(np.uint8)).newbyteorder(">")
    # imgs = imgs.reshape((size,nrows,ncols))
    # Flattened
    imgs = imgs.reshape((size, nrows * ncols))


imgs = (imgs / 255).astype("float32")

with open("labels", "rb") as i:
    magic, size = struct.unpack(">II", i.read(8))
    labs = np.fromfile(i, dtype=np.dtype(np.uint8)).newbyteorder(">")


#Reduced size for testing
size=10000
imgs=imgs[50000:]
labs=labs[50000:]
#print(imgs.shape)

# To categorial: one-hot matrix:
cats = np.zeros((size, 10))
for i in range(len(labs)):
    vl = labs[i]
    cats[i][vl] = 1

print("categorials done")


##At this stage, imgs is 60000*784 pixel images between 0 and 1
# cats is categorical one-hot encoded vectors

# Get a set of unique random integers to decide train/test split
rndintegers = random.sample(range(size), k=int(trainsize * size))
# print("tapa")


randindices = list(range(size))
trainN = int(trainsize * size)
np.random.shuffle(randindices)
train_indices = randindices[:trainN]
test_indices = randindices[trainN:]

trainimages, testimages = imgs[train_indices], imgs[test_indices]
trainlabels, testlabels = labs[train_indices], labs[test_indices]

hidden_1 = 3
input_layer = 9
a1 = np.random.randn(hidden_1, input_layer) * np.sqrt(1.0 / hidden_1)
# print(a1)



#sys.exit()


network = DeepNeuralNetwork(sizes=[784, 128, 64, 10])
network.train(trainimages,trainlabels,testimages,testlabels)
