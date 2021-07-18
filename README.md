# NNscratch
Creating an MNIST NN from scratch without any external libraries except NumPy. I will be doing a few different optimizators, activation functions and all that. 


Would also be cool to see if I could GA the optimal net dimensions just for funs!


Best accuracy so far is **0.91967** with parameters:

   - sizes = [784, 256, 128, 96, 64, 32, 10]
   - varGamma = True
   - decreasing = True
   - epochs = 20
   - gamma = 0.01
   - batchSize = 32
   - optimizer = "SGD"
   - activation = "leakyRELU"
   - used 0.4 of the whole dataset




# Todo

- ADAM: some ideas are there, but it's not really working yet.
- Config file: duh
- Comments: clear for others to understand


# File contents

- `nnscratch.py` contains all the helper functions and setup functions for the net 
- `dnn.py` class for the deep neural net with all its methods
- `images` MNIST images in binary format
- `labels` MNIST labels in binary format


# Structure

Structure is adabtable! Just change the network creation parameters in the file `nnscratch.py`, on line 115. The input layer and the output layers are stable, everything else can be changed freely. Bigger nets naturally take up more computation time, though.

 - Input layer: flattened 28x28=784
 - Output layer: reduce to last 10



# Usage

First, get the dataset from `http://yann.lecun.com/exdb/mnist/`

Then make sure that the files containing images and labels are in the same folder as the python scripts. Name them however you want, as long as you remember to change the parameters on line 26 of `nnscratch.py` to match.

The default train/test size split is 80/20. If you wish to modify that, change the line 68 of `nnscratch.py`.

One can use a smaller chunk of the whole dataset by providing a float between 0 and 1 as a command line argument. If no argument is given, the whole dataset will be used.


All the parameters for the net itself are defined starting from like 115 of `nnscratch.py`.

 - `sizes` is the configuration of the network layers. Use whatever configuration you want, as long as the first and the last layer are 784 and 10 respectively. Default is [784, 128, 64, 32, 10]
 - `varGamma` governs if the learning rate is adjusted between each epoch to help the convergence of the net. Default is False
 - `decreasing` governs if the variable learning rate starts small and increases or vice versa. Default is False
 - `epochs` how many epochs do we want to run. Default is 25
 - `batchSize` if using mini batch gradient descent, this depicts the batch size. Default is 32
 - `optimizer` selects which optimizer to use. Right now, only Stochastic Gradient Descent ("SGD") and Mini Batch Gradient Descent ("MBGD") are implemented. Default is SGD
 - `activation` selects the activation function to use. Currently the options are RELU ("RELU"), leaky RELU ("leakyRELU") and sigmoid ("sigmoid"). Default is leaky RELU

