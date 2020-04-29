import numpy as np


class NeuralNetwork():

    def __init__(self, a):
        np.random.seed(5)
        self. synaptic_weights = 2 * np.random.random((a, 1)) -1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivation(self, x):
        return x * (1 - x)
        

    def train(self, training_in, training_out, training_iterations, l_rate):
        for iteration in range(training_iterations):
            output = self.think(training_in)
            error = training_out - output
            adjustments = np.dot(training_in.T, error * self.sigmoid_derivation(output))
            self.synaptic_weights += adjustments * l_rate

    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output