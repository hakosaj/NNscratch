import numpy as np
import time


class DeepNeuralNetwork:

    # Input layer: flattened 28x28=784
    # First hidden layer: reduce to 16x16=256
    # Second hidden layer: reduce to 8x8 = 64
    # Output layer: reduce to last 10

    def __init__(self, sizes, epochs=10, gamma=0.001):
        self.sizes = sizes
        self.epochs = epochs
        self.gamma = gamma
        self.params = self.initializeNet()
        print(f"Initialized network with layers as {self.sizes}")

    def initializeNet(self):

        # initialize the sizes for the NN layers
        input, hiddenOne, hiddenTwo, output = (
            self.sizes[0],
            self.sizes[1],
            self.sizes[2],
            self.sizes[3],
        )

        # Weight vectors!
        params = {
            "W1": np.random.randn(hiddenOne, input) * np.sqrt(1.0 / hiddenOne),
            "W2": np.random.randn(hiddenTwo, hiddenOne) * np.sqrt(1.0 / hiddenTwo),
            "W3": np.random.randn(output, hiddenTwo) * np.sqrt(1.0 / output),
        }

        return params

    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x)) / ((np.exp(-x) + 1) ** 2)
        return 1 / (1 + np.exp(-x))

    def softmax(self, x, derivative=False):
        exps = np.exp(x - x.max())
        if derivative:
            return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)

    def forwardPass(self, x_train):
        params = self.params

        # input layer activations becomes sample
        params['A0'] = x_train

        # input layer to hidden layer 1
        params['Z1'] = np.dot(params["W1"], params['A0'])
        params['A1'] = self.sigmoid(params['Z1'])

        # hidden layer 1 to hidden layer 2
        params['Z2'] = np.dot(params["W2"], params['A1'])
        params['A2'] = self.sigmoid(params['Z2'])

        # hidden layer 2 to output layer
        params['Z3'] = np.dot(params["W3"], params['A2'])
        params['A3'] = self.softmax(params['Z3'])

        return params['A3']

    def backprop(self, y_train, output):

        params = self.params
        change_w = {}

        # Calculate W3 update
        error = 2 * (output - y_train) / output.shape[0] * self.softmax(params['Z3'], derivative=True)
        change_w['W3'] = np.outer(error, params['A2'])

        # Calculate W2 update
        error = np.dot(params['W3'].T, error) * self.sigmoid(params['Z2'], derivative=True)
        change_w['W2'] = np.outer(error, params['A1'])

        # Calculate W1 update
        error = np.dot(params['W2'].T, error) * self.sigmoid(params['Z1'], derivative=True)
        change_w['W1'] = np.outer(error, params['A0'])

        return change_w

    def train(self, x_train, y_train, x_val, y_val):
        print(f"Starting network training with {self.epochs} epochs")
        start_time = time.time()
        for iteration in range(self.epochs):
            i = 0
            for x, y in zip(x_train, y_train):
                i += 1
                if i % 2000 == 0:
                    print(f"done {i}/{len(x_train)}")
                self.updateNetworkParameters(self.backprop(y, self.forwardPass(x)))

            acc = self.computeAccuracy(x_val, y_val)
            print(
                "Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2}".format(
                    iteration + 1, time.time() - start_time, acc
                )
            )

    def updateNetworkParameters(self, changes_to_w):

        for key, value in changes_to_w.items():
            self.params[key] -= self.gamma * value

    def computeAccuracy(self, x_val, y_val):

        preds = []

        for x, y in zip(x_val, y_val):
            pred = np.argmax(self.forwardPass(x))
            preds.append(pred == np.argmax(y))

        return np.mean(preds)
