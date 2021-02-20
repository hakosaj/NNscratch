# NNscratch
Creating an MNIST NN from scratch without any external libraries except NumPy. The basic structure is there, sometimes it works and sometimes not. I wonder how to go from here

https://mlfromscratch.com/neural-network-tutorial/#/

http://yann.lecun.com/exdb/mnist/

# Structure

 - Input layer: flattened 28x28=784
 - First hidden layer: reduce to 128
 - Second hidden layer: reduce to 64
 - Output layer: reduce to last 10


 # Todo

  - Testing the net:for some reason 0 acc sometimes. Is there a typo somewhere?
  - Write everything open: what is actually happening in each step?
  - Relu activation function instead of a sigmoid
  - Use of biases: add them to Z before actication in fw pass, and update in backpass. Careful with dims!
  - FW pass and BW pass to loops so easier to modify and maintain.
  - Change the initi function that sizes can be modified easier.
  - Mini batc gradient descent: no update each sample, but update by average of gradients of the batch (size less tha 64)
  - Implement ADAM. Momentum, adaptive learning rate-> adam 
  - GA opt of layer sizes?
  - Visualization!