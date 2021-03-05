import numpy as np
import time
import math
import collections


class DeepNeuralNetwork:

    # Input layer: flattened 28x28=784
    # First hidden layer: reduce to 16x16=256
    # Second hidden layer: reduce to 8x8 = 64
    # Output layer: reduce to last 10

    def __init__(
        self, sizes, epochs=12, gamma=0.08, variableGamma=False, decreasing=False
    ):
        self.sizes = sizes
        self.epochs = epochs
        self.gamma = gamma
        self.variableGamma = variableGamma
        self.decreasing = decreasing
        self.params = self.initializeVariableNet()

        if variableGamma:
            if decreasing:
                self.gamma = 0.2
            else:
                self.gamma = 0.0001
        self.losses = []
        self.gammas = []
        print(f"Initialized network with layers as {self.sizes}")

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
        print(f"Starting network training with {self.epochs} epochs")
        start_time = time.time()
        for iteration in range(self.epochs):
            i = 0
            for x, y in zip(x_train, y_train):
                i += 1
                # if i % 2000 == 0:
                # print(f"done {i}/{len(x_train)}")
                # self.variableForwardPass(x)
                a, b = self.variableBackprop(y, self.variableForwardPass(x))
                self.updateNetworkParameters(a, b)

            acc, loss = self.computeAccuracy(x_val, y_val)
            print(
                "Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2},".format(
                    iteration + 1, time.time() - start_time, acc
                ),
                end=" ",
            )
            print(f"loss: {loss}")

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
