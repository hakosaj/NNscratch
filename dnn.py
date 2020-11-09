class DeepNeuralNetwork:

    def __init__(self,sizes, epochs=10, gamma=0.001):
        self.sizes=sizes
        self.epochs=epochs
        self.gamma=gamma
        self.params=self.initializeNet()


    def initializeNet(self):

        input,hiddenOne,hiddenTwo,output=self.sizes[0],self.sizes[1],self.sizes[2],self.sizes[3]
        

    