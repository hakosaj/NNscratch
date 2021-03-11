import numpy as np
import time
import math
import collections
from tqdm import tqdm


class DeepNeuralNetwork:

    # Input layer: flattened 28x28=784
    # First hidden layer: reduce to 16x16=256
    # Second hidden layer: reduce to 8x8 = 64
    # Output layer: reduce to last 10

    def __init__(
        self,
        sizes,
        epochs=15,
        gamma=0.01,
        batchSize=20,
        variableGamma=False,
        decreasing=False,
        optimizer="SGD",
        activation="LeakyRELU",
    ):
        self.sizes = sizes
        self.epochs = epochs
        self.gamma = gamma
        self.variableGamma = variableGamma
        self.decreasing = decreasing
        self.params = self.initializeVariableNet()
        self.optimizer = optimizer
        self.batchSize = batchSize
        self.activation = activation

        # ADAM parameters
        if optimizer == "ADAM":
            self.beta1 = 0.9
            self.beta2 = 0.99
            self.epsilon = 0.00000001
            self.m_dw, self.v_dw = 0, 0
            self.m_db, self.v_db = 0, 0

        if variableGamma:
            if decreasing:
                self.gamma = 0.3
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

    def softmax(self, x, derivative=False):
        exps = np.exp(x - x.max())
        if derivative:
            return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)

    def forwardPass2(self, x_train):
        params = self.params

        # input layer activations becomes sample
        params["A0"] = x_train

        # input layer to hidden layer 1
        params["Z1"] = np.dot(params["W1"], params["A0"])
        params["Z1"] += params["B1"]
        params["A1"] = self.sigmoid(params["Z1"])

        # hidden layer 1 to hidden layer 2
        params["Z2"] = np.dot(params["W2"], params["A1"])
        params["Z2"] += params["B2"]
        params["A2"] = self.sigmoid(params["Z2"])

        # hidden layer 2 to output layer
        params["Z3"] = np.dot(params["W3"], params["A2"])
        params["Z3"] += params["BO"]
        params["A3"] = self.softmax(params["Z3"])

        return params["A3"]

    def variableForwardPass(self, x_train):
        params = self.params

        params["A0"] = x_train
        # All layers except output
        for i in range(0, len(self.sizes) - 2):
            params[f"Z{i+1}"] = np.dot(params[f"W{i+1}"], params[f"A{i}"])
            params[f"Z{i+1}"] += params[f"B{i+1}"]
            params[f"A{i+1}"] = self.leakyRelu(params[f"Z{i+1}"])

        # Output
        last = len(self.sizes) - 1
        params[f"Z{last}"] = np.dot(params[f"W{last}"], params[f"A{last-1}"])
        params[f"Z{last}"] += params["BO"]
        params[f"A{last}"] = self.softmax(params[f"Z{last}"])

        return params[f"A{last}"]

    def forwardPass(self, x_train):
        params = self.params

        # input layer activations becomes sample
        params["A0"] = x_train

        # input layer to hidden layer 1
        params["Z1"] = np.dot(params["W1"], params["A0"])
        params["Z1"] += params["B1"]
        params["A1"] = self.leakyRelu(params["Z1"])

        # hidden layer 1 to hidden layer 2
        params["Z2"] = np.dot(params["W2"], params["A1"])
        params["Z2"] += params["B2"]
        params["A2"] = self.leakyRelu(params["Z2"])

        # hidden layer 2 to output layer
        params["Z3"] = np.dot(params["W3"], params["A2"])
        params["Z3"] += params["BO"]
        params["A3"] = self.softmax(params["Z3"])

        return params["A3"]

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

    def backprop(self, y_train, output):

        params = self.params
        change_w = {}
        change_b = {}

        # Calculate W3 update
        error = (
            2
            * (output - y_train)
            / output.shape[0]
            * self.softmax(params["Z3"], derivative=True)
        )
        change_w["W3"] = np.outer(error, params["A2"])
        change_b["BO"] = error

        # Calculate W2 update
        error = np.dot(params["W3"].T, error) * self.sigmoid(
            params["Z2"], derivative=True
        )
        change_w["W2"] = np.outer(error, params["A1"])
        change_b["B2"] = error

        # Calculate W1 update
        error = np.dot(params["W2"].T, error) * self.sigmoid(
            params["Z1"], derivative=True
        )
        change_w["W1"] = np.outer(error, params["A0"])
        change_b["B1"] = error

        return change_w, change_b

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
                        self.updateNetworkParameters(a, b)
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
                            self.updateNetworkParameters(ws, bs)
                            passflag = False
                            g = 0
                    pbar.update()

            acc, loss = self.computeAccuracy(x_val, y_val)
            print(
                "Epoch: {0}, Time Spent: {1:.2f}s, Learning rate: {2:.4f},".format(
                    iteration + 1, time.time() - start_time, self.gamma
                ),
            )
            print("Accuracy: {0:.5f}, Loss: {1:.5f}\n\n".format(acc, loss))
            # print(f"loss: {loss}\n\n")

            if self.variableGamma:
                if not self.decreasing:
                    self.gamma = self.gamma ** 0.85
                else:
                    self.gamma = self.gamma ** 1.15

        print("Training done")
        return self.gammas, self.losses

    def updateNetworkParameters(self, changes_to_w, changes_to_b):
        for key, value in changes_to_w.items():
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
