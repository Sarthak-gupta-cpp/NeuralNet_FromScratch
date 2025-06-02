import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

train_data, validation_data = train_test_split(train_data)
train_y = train_data["label"].to_numpy()
train_x = train_data.drop(columns=["label"]).to_numpy()

validation_x = validation_data.drop(columns=["label"]).to_numpy()
validation_y = validation_data["label"].to_numpy()


class Neuron:
    def __init__(self, input_shape):
        self.inputs = input_shape
        self.weights = np.random.randn(input_shape) * 0.01
        self.bias = np.random.randn() * 0.01

    def forward(self, x):
        return np.dot(x, self.weights) + self.bias


class Linear_Layer:
    def __init__(self, input_shape, output_shape, activation_fn=lambda x: x):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.activation_fn = activation_fn
        # self.neurons = np.array([Neuron(input_shape) for i in range(output_shape)])  (making a list of neurons was not efficient and i was not able to include batches using this)
        self.weights = np.random.randn(input_shape, output_shape) * 0.01
        self.bias = np.random.randn(output_shape)

    def forward(self, x):
        # return self.activation_fn(np.array([i.forward(x) for i in self.neurons]))
        return self.activation_fn(np.matmul(x, self.weights) + self.bias)


class NeuralNet:
    def __init__(self, input_shape=784, output_shape=10, hidden_layer=64):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.input_layer = Linear_Layer(
            input_shape=input_shape, output_shape=hidden_layer, activation_fn=ReLu
        )
        self.hidden_layer1 = Linear_Layer(
            input_shape=hidden_layer, output_shape=hidden_layer, activation_fn=ReLu
        )
        self.output_layer = Linear_Layer(
            input_shape=hidden_layer, output_shape=output_shape, activation_fn=ReLu
        )

    def forward(self, x):
        y = self.input_layer.forward(x)
        y = self.hidden_layer1.forward(y)
        y = self.output_layer.forward(y)
        y = Softmax(y)
        return y


def Softmax(x):
    # This implementation was without accomadating for batches
    # m = np.max(x)  # to handle overflow when x has large values
    # sum = np.sum(np.exp(x - m))
    # return np.exp(x - m) / sum

    # With Batches
    m = np.max(x, axis=1, keepdims=True)
    sum = np.sum(np.exp(x - m), axis=1, keepdims=True)
    return np.exp(x - m) / sum


def ReLu(x):
    return np.maximum(x, 0)


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def MSE_loss_fn(y_preds, y_actual):
    sum = np.mean((y_preds.flatten() - y_actual.flatten()) ** 2)
    sum = sum / int(y_preds.flatten().size)


def Cross_Entropy_loss():
    return


def modify_y(y, num_classes):
    new_y = np.zeroes((y.size, num_classes))
    for i in y.size:
        new_y[i][y[i]] = 1
    return new_y


model = NeuralNet()
model.forward(train_x[0:10])
