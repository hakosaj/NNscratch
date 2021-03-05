import numpy as np
import struct
import random
import matplotlib.pyplot as plt
from dnn import DeepNeuralNetwork
import time
import sys

# https://mlfromscratch.com/neural-network-tutorial/#/
trainsize = 0.8
testsize = 1 - trainsize




plotting=True

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


# Reduced size for testing
try:
    sz=int(round(float(sys.argv[1])*len(labs)))
    size=sz
    imgs=imgs[60000-sz:]
    labs=labs[60000-sz:]
except IndexError:
    sz=0
    
    pass

# print(imgs.shape)

# To categorial: one-hot matrix:
cats = np.zeros((size, 10))
for i in range(len(labs)):
    vl = labs[i]
    cats[i][vl] = 1

#print("categorials done")
print(f"Using {len(labs) if sz==0 else size} samples")


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
trainlabels, testlabels = cats[train_indices], cats[test_indices]

hidden_1 = 3
input_layer = 9
a1 = np.random.randn(hidden_1, input_layer) * np.sqrt(1.0 / hidden_1)
# print(a1)


# sys.exit()


#network = DeepNeuralNetwork(sizes=[784, 128, 64, 10],variableGamma=True)
varGamma=True
decreasing=True
network = DeepNeuralNetwork(sizes=[784, 128, 64, 10],variableGamma=varGamma,decreasing=decreasing)
gammas,losses=network.train(trainimages, trainlabels, testimages, testlabels)


#Plot the losses and learning rates, only if adaptive learning is used!
if plotting:
    if varGamma:
        if decreasing:
            plt.plot(gammas,losses)
            plt.xscale("log")
            plt.axis([max(gammas)*1.1 , min(gammas)*1.1,min(losses)*1.01, max(losses)*0.9])
            plt.xlabel("learning rate")
            plt.ylabel("loss")
            plt.show()
        else:
            plt.plot(gammas,losses)
            plt.xscale("log")
            plt.axis([min(gammas), max(gammas),max(losses), min(losses)])
            plt.xlabel("learning rate")
            plt.ylabel("loss")
            plt.show()