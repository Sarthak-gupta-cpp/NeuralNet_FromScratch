import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    def __init__(self, input_shape):
        self.inputs = input_shape
        self.weights = np.random.randn(input_shape)
        self.bias = np.random.randn()

    def forward(self, x):
        return np.dot(x, self.weights) + self.bias


class Linear_Layer:
    def __init__(self, input_shape, output_shape, activation_fn=lambda x: x):
        self.input_shape = input_shape
        self.output_shape = self.output_shape
        self.activation_fn = activation_fn
        self.neurons = np.array([Neuron(input_shape) for i in range(output_shape)])

    def forward(self, x):
        return self.activation_fn(np.array([i.forward(x) for i in self.neurons]))


class NeuralNet:
    def __init__(self, input_shape=784, output_shape=10):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.input_layer = Linear_Layer(
            input_shape=input_shape, output_shape=64, activation_fn=ReLu
        )
        self.hidden_layer1 = Linear_Layer(
            input_shape=64, output_shape=64, activation_fn=ReLu
        )
        self.output_layer = Linear_Layer(
            input_shape=64, output_shape=10, activation_fn=ReLu
        )

    def forward(self, x):
        y = self.input_layer.forward(x)
        y = self.hidden_layer1.forward(y)
        y = self.output_layer.forward(y)
        y = Softmax(y)
        return y


def Softmax(x):
    sum = np.sum(np.exp(x))
    return np.exp(x) / sum


def ReLu(x):
    return np.maximum(x, 0)


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
