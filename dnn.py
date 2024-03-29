# Jussi Hakosalo - hakosaj - 2021


import numpy as np
import time
import math
import collections
from tqdm import tqdm

#https://mlfromscratch.com/neural-network-tutorial/
#https://github.com/addyg/Neural-Network-from-Scratch/blob/master/02%20SRC/NeuralNetwork.py
class DeepNeuralNetwork:

    # Input layer: flattened 28x28=784
    # First hidden layer: reduce to 16x16=256
    # Second hidden layer: reduce to 8x8 = 64
    # Output layer: reduce to last 10

    def __init__(
        self,
        sizes=[784, 128, 64, 32, 10],
        epochs=25,
        gamma=0.01,
        batchSize=32,
        variableGamma=False,
        decreasing=False,
        optimizer="SGD",
        activation="leakyRELU",
    ):
        self.sizes = sizes
        self.epochs = epochs
        self.gamma = gamma
        self.variableGamma = variableGamma
        self.decreasing = decreasing
        self.optimizer = optimizer
        self.batchSize = batchSize
        self.dropreduce=False
        self.activation = activation

        self.moments={}
        self.vmoments={}
        # ADAM parameters
        if optimizer == "SGD":
            #ADAM STUFF
            self.beta1 = 0.9
            self.beta2 = 0.99
            self.epsilon = 0.00000001

        self.params = self.initializeVariableNet()
        if variableGamma:
            if decreasing:
                self.gamma = 0.5
            else:
                self.gamma = 0.0001
        self.losses = []
        self.gammas = []
        self.printNetworkInfo()

    def printNetworkInfo(self):
        print(f"\nInitialized network with layers as {self.sizes}")
        print(f"Using {self.optimizer} as optimizer,")
        print(f"{self.activation} as activation")
        if self.variableGamma:
            print(f"Using an adaptive learning rate, starting from {self.gamma}")
        else:
            print(f"Using a static learning rate of {self.gamma}")
        print(f"Starting network training with {self.epochs} epochs\n")

    def initializeVariableNet(self):
        # Input layer and output layer always exist
        layers = self.sizes

        params = {}
        for i in range(0, len(layers) - 1):
            params[f"W{i+1}"] = np.random.randn(layers[i + 1], layers[i]) * np.sqrt(
                1.0 / layers[i + 1]
            )

        for i in range(0, len(layers) - 2):
            params[f"B{i+1}"] = np.random.randn(layers[i + 1]) * np.sqrt(
                1.0 / layers[i + 1]
            )
        params[f"BO"] = np.random.randn(layers[-1]) * np.sqrt(1.0 / layers[-1])


        #ADAM
    
        for i in range(0, len(layers) - 1):
            self.moments[f"W{i+1}"] = np.zeros((layers[i + 1], layers[i]))

        for i in range(0, len(layers) - 1):
            self.vmoments[f"W{i+1}"] = np.zeros((layers[i + 1], layers[i]))

        return params

    def initializeNet(self):

        # number of nodes in each layer
        input_layer = self.sizes[0]
        hidden_1 = self.sizes[1]
        hidden_2 = self.sizes[2]
        output_layer = self.sizes[3]

        params = {
            "W1": np.random.randn(hidden_1, input_layer) * np.sqrt(1.0 / hidden_1),
            "W2": np.random.randn(hidden_2, hidden_1) * np.sqrt(1.0 / hidden_2),
            "W3": np.random.randn(output_layer, hidden_2) * np.sqrt(1.0 / output_layer),
            "B1": np.random.randn(hidden_1) * np.sqrt(1.0 / hidden_1),
            "B2": np.random.randn(hidden_2) * np.sqrt(1.0 / hidden_2),
            "BO": np.random.randn(output_layer) * np.sqrt(1.0 / output_layer),
        }

        return params

    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x)) / ((np.exp(-x) + 1) ** 2)
        return 1 / (1 + np.exp(-x))

    def leakyRelu(self, x, derivative=False):
        if derivative:
            return np.full((x.shape), 1) if x > 0 else np.full((x.shape), 0.01)
        return np.array(list(map(lambda f: f if f > 0 else f * 0.01, x)))

    def relu(self, x, derivative=False):
        if derivative:
            return np.full((x.shape), 1) if x > 0 else np.full((x.shape), 0)
        return np.array(list(map(lambda f: f if f > 0 else 0, x)))

    def softmax(self, x, derivative=False):
        exps = np.exp(x - x.max())
        if derivative:
            return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)

    def variableForwardPass(self, x_train):
        params = self.params

        params["A0"] = x_train
        # All layers except output
        for i in range(0, len(self.sizes) - 2):
            params[f"Z{i+1}"] = np.dot(params[f"W{i+1}"], params[f"A{i}"])
            params[f"Z{i+1}"] += params[f"B{i+1}"]

            if self.activation == "leakyRELU":
                params[f"A{i+1}"] = self.leakyRelu(params[f"Z{i+1}"])
            elif self.activation == "RELU":
                params[f"A{i+1}"] = self.relu(params[f"Z{i+1}"])
            elif self.activation == "sigmoid":
                params[f"A{i+1}"] = self.sigmoid(params[f"Z{i+1}"])
            else:
                params[f"A{i+1}"] = self.leakyRelu(params[f"Z{i+1}"])

        # Output
        last = len(self.sizes) - 1
        params[f"Z{last}"] = np.dot(params[f"W{last}"], params[f"A{last-1}"])
        params[f"Z{last}"] += params["BO"]
        params[f"A{last}"] = self.softmax(params[f"Z{last}"])

        return params[f"A{last}"]

    def variableBackprop(self, y_train, output):
        params = self.params
        change_w = {}
        change_b = {}

        # Last update
        last = len(self.sizes) - 1
        error = (
            2
            * (output - y_train)
            / output.shape[0]
            * self.softmax(params[f"Z{last}"], derivative=True)
        )
        change_w[f"W{last}"] = np.outer(error, params[f"A{last-1}"])
        change_b["BO"] = error

        for i in reversed(range(1, last)):
            error = np.dot(params[f"W{i+1}"].T, error) * self.sigmoid(
                params[f"Z{i}"], derivative=True
            )
            change_w[f"W{i}"] = np.outer(error, params[f"A{i-1}"])
            change_b[f"B{i}"] = error

        return change_w, change_b

    def printResults(self, iteration, start_time, acc, loss):
        print(
            "Epoch: {0}, Time Spent: {1:.2f}s, Learning rate: {2:.5f}".format(
                iteration + 1, time.time() - start_time, self.gamma
            ),
        )
        print("Accuracy: {0:.5f}, Loss: {1:.5f}\n\n".format(acc, loss))

    def train(self, x_train, y_train, x_val, y_val):
        start_time = time.time()
        for iteration in range(self.epochs):
            i = 0

            if self.optimizer == "MBGD":
                g = 0
                batchSize = self.batchSize
                ws = {}
                bs = {}
                passflag = False

            # for x, y in tqdm(zip(x_train, y_train)):
            trainiters = zip(x_train, y_train)
            with tqdm(total=len(x_train)) as pbar:
                for x, y in trainiters:
                    i += 1

                    a, b = self.variableBackprop(y, self.variableForwardPass(x))

                    if self.optimizer == "SGD":
                        self.updateNetworkParameters(a, b,i)
                    if self.optimizer == "MBGD":
                        if not passflag:
                            ws = a
                            bs = b
                            passflag = True
                        else:
                            for key, value in a.items():
                                ws[key] += value
                            for key, value in b.items():
                                bs[key] += value
                        g += 1
                        if g == batchSize:
                            ws = {k: v / batchSize for k, v in ws.items()}
                            bs = {k: v / batchSize for k, v in bs.items()}
                            self.updateNetworkParameters(ws, bs,i)
                            passflag = False
                            g = 0
                    pbar.update()

            acc, loss = self.computeAccuracy(x_val, y_val)
            self.printResults(iteration, start_time, acc, loss)

            if self.variableGamma:
                if not self.decreasing:
                    self.gamma = self.gamma ** 0.85
                else:
                    self.gamma = self.gamma ** 1.15


            if iteration <4 and acc>0.88 and not self.dropreduce:
                self.gamma=self.gamma/1.5
                self.dropreduce=True
                print("Drop reduced the learning rate\n")

        print("Training done")
        return self.gammas, self.losses

    def updateNetworkParameters(self, changes_to_w, changes_to_b,i):
        for key, value in changes_to_w.items():


            #g(t)=f'(x(t-1))=derivative
            derivative=value

            #first moment
            #m(t)=beta1*m(t-1)+(1-beta1)*g(t)
            self.moments[key]=self.moments[key]*self.beta1+(1-self.beta1)*derivative

            #Bias correct first moment
            self.moments[key]=self.moments[key]/(1-self.beta1)

            #Decay
            self.beta1**(i+1)
            
            


            #second moment
            #v(t)=beta2*v(t-1)+(1-beta2)*g(t)^2
            self.vmoments[key]=self.beta2*self.vmoments[key]+(1-self.beta2)*(derivative**2)
            
            #Bias correct the first moment
            self.vmoments[key]=self.vmoments[key]/(1-self.beta2)

            #Decay
            self.beta2**(i+1)


            #ADAM
            self.params[key]-=self.gamma*self.moments[key]/(self.vmoments[key]**(0.5)+self.epsilon)

            #Classic gradient desc
            self.params[key] -= self.gamma * value


        for key, value in changes_to_b.items():
            self.params[key] -= self.gamma * value

    def computeAccuracy(self, x_val, y_val):

        preds = []
        # outputs=[]
        logloss = 0

        for x, y in zip(x_val, y_val):
            outp = self.variableForwardPass(x)
            pred = np.argmax(outp)
            preds.append(pred == np.argmax(y))
            logloss += np.dot(y, np.log(outp))
        self.gammas.append(self.gamma)
        self.losses.append(logloss)

        return np.mean(preds), logloss
