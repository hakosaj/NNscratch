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
            "W1": np.random.randn(hidden_1, input_layer) * np.sqrt(1.0 / hidden_1),
            "W2": np.random.randn(hidden_2, hidden_1) * np.sqrt(1.0 / hidden_2),
            "W3": np.random.randn(output_layer, hidden_2) * np.sqrt(1.0 / output_layer),
        }

        return params

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def dSigmoid(self, x):
        return (np.exp(-x)) / ((np.exp(-x) + 1) ** 2)

    def sotfmax(self, x):
        es = np.exp(x - x.max())
        return es / np.sum(es, axis=0)

    def dSoftmax(self,x):


    def fowardPass(self, x_train):
        params = self.params

        # Input layer activation-> first weights
        params["A0"] = x_train

        # input layer to hidden layer 1
        params["Z1"] = np.dot(params["W1"], params["A0"])
        params["A1"] = self.sigmoid(params["Z1"])

        # hidden layer 1 to hidden layer 2
        params["Z2"] = np.dot(params["W2"], params["A1"])
        params["A2"] = self.sigmoid(params["Z2"])

        # hidden layer 2 to output layer
        params["Z3"] = np.dot(params["W3"], params["A2"])
        params["A3"] = self.softmax(params["Z3"])

        return params["A3"]

    def backprop(self, y_train, output):

        params = self.params()
        w_changes = {}

        #Update to the outermost layer
        #Is there an error here?
        error = 2* (output-y_train)/output.shape[0]*self.dSoftmax(['Z3'])
        w_changes['W3'] = np.outer(error,params['A2'])

        #Moving inwards
        # Calculate W2 update
        error = np.dot(params['W3'].T, error) * self.sigmoid(params['Z2'], derivative=True)
        w_changes['W2'] = np.outer(error, params['A1'])

        # Calculate W1 update
        error = np.dot(params['W2'].T, error) * self.sigmoid(params['Z1'], derivative=True)
        w_changes['W1'] = np.outer(error, params['A0'])

        return w_changes

