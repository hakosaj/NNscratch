# Jussi Hakosalo - hakosaj - 2021


# The best library ever, all linear algebra is done with this baby
import numpy as np

# For unpacking the byte format file the dataset comes in
import struct

# Randomness breh
import random

# Cool graphs are cool
import matplotlib.pyplot as plt

# The actual network class
from dnn import DeepNeuralNetwork

# Time stuff
import time

# Panic exit if necessary
import sys


def unpackImages(imagefile="images", labelfile="labels"):

    """
    Helper function to extract the data from the original MNIST dataset, which is given in byte format.

    Parameters
    ----------
    imagefile : str
        File containing the byteform images
    labelfile : str
        File containing the byteform labels

    Returns
    -------
    imgs: numpy.ndarray
        Numpy array containing the images scaled down and reshaped
    labs: numpy.ndarray
        Numpy array containing the labels


    """

    # Struct unpack magic happening here
    with open(imagefile, "rb") as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        imgs = np.fromfile(f, dtype=np.dtype(np.uint8)).newbyteorder(">")

        # Flatten the images from 28x28 to a 784-vector
        imgs = imgs.reshape((size, nrows * ncols))

    # Normalize the pixels
    imgs = (imgs / 255).astype("float32")

    with open(labelfile, "rb") as i:
        magic, size = struct.unpack(">II", i.read(8))
        labs = np.fromfile(i, dtype=np.dtype(np.uint8)).newbyteorder(">")

    return imgs, labs


def net():
    trainsize = 0.8
    plotting = True
    imgs, labs = unpackImages()

    try:
        sz = int(round(float(sys.argv[1]) * len(labs)))
        if sz > 60000:
            sz = 60000
        size = sz
        imgs = imgs[60000 - sz :]
        labs = labs[60000 - sz :]
    except IndexError:
        sz = 0
        pass

    # To categorial: one-hot matrix:
    cats = np.zeros((size, 10))
    for i in range(len(labs)):
        vl = labs[i]
        cats[i][vl] = 1

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

    # Define NN parameters here
    sizes = [784, 128, 96, 64, 40, 32, 10]
    varGamma = True
    decreasing = True
    epochs = 20
    gamma = 0.01
    batchSize = 32
    optimizer = "SGD"
    activation = "leakyRELU"

    network = DeepNeuralNetwork(
        sizes=sizes,
        epochs=epochs,
        gamma=gamma,
        batchSize=batchSize,
        variableGamma=varGamma,
        decreasing=decreasing,
        optimizer=optimizer,
        activation=activation,
    )
    gammas, losses = network.train(trainimages, trainlabels, testimages, testlabels)

    # Plot the losses and learning rates, only if adaptive learning is used!
    if plotting:
        if varGamma:
            if decreasing:
                plt.plot(gammas, losses)
                plt.xscale("log")
                plt.axis(
                    [
                        max(gammas) * 1.1,
                        min(gammas) * 1.1,
                        min(losses) * 1.01,
                        max(losses) * 0.9,
                    ]
                )
                plt.xlabel("learning rate")
                plt.ylabel("loss")
                plt.show()
            else:
                plt.plot(gammas, losses)
                plt.xscale("log")
                plt.axis([min(gammas), max(gammas), max(losses), min(losses)])
                plt.xlabel("learning rate")
                plt.ylabel("loss")
                plt.show()


net()
