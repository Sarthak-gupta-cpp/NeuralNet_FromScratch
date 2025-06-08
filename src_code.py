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
        self.z = 0
        self.a = 0
        # self.neurons = np.array([Neuron(input_shape) for i in range(output_shape)])  (making a list of neurons was not efficient and i was not able to include batches using this)
        self.weights = np.random.randn(input_shape, output_shape) * 0.01
        self.bias = np.random.randn(output_shape) * 0.01

    def forward(self, x):
        # return self.activation_fn(np.array([i.forward(x) for i in self.neurons]))
        self.z = np.matmul(x, self.weights) + self.bias
        self.a = self.activation_fn(self.z)
        return self.a


class NeuralNet:
    def __init__(
        self,
        input_shape=784,
        output_shape=10,
        hidden_layer=64,
        learning_rate=0.001,
        activation_fn=lambda x: x,
    ):
        self.act = activation_fn
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learning_rate = learning_rate
        self.input_layer = Linear_Layer(
            input_shape=input_shape,
            output_shape=hidden_layer,
            activation_fn=activation_fn,
        )
        self.hidden_layer1 = Linear_Layer(
            input_shape=hidden_layer,
            output_shape=hidden_layer,
            activation_fn=activation_fn,
        )
        self.output_layer = Linear_Layer(
            input_shape=hidden_layer, output_shape=output_shape
        )

    def forward(self, x):
        y = self.input_layer.forward(x)
        y = self.hidden_layer1.forward(y)
        y = self.output_layer.forward(y)
        y = Softmax(y)
        return y

    def backpropogation(self, y_actual, y_preds, train_x):
        dl_dz = (
            (y_preds - y_actual) / y_preds.shape[0]
        )  # (32x10)#dividing by batch size to get normalized loss (do not scale with batch_size)

        # for output layer 64 -> 10
        #          (64x32)               (32x10)
        dl_dw = (
            self.hidden_layer1.a.T @ dl_dz
        )  #  (64x10) #@ is for matmul, in case i forget
        dl_db = dl_dz.sum(axis=0)  # (1, 10)
        dl_da = dl_dz @ self.output_layer.weights.T  # (32x64)

        self.output_layer.weights = (
            self.output_layer.weights - self.learning_rate * dl_dw
        )
        self.output_layer.bias = self.output_layer.bias - self.learning_rate * dl_db

        # for hidden_layer1 64 -> 64 -> relu -> a
        dl_dz = dl_da * derivative_ReLu(self.hidden_layer1.a)  # (32x64)
        dl_dw = self.input_layer.a.T @ dl_dz
        dl_db = dl_dz.sum(axis=0)
        dl_da = dl_dz @ self.hidden_layer1.weights.T

        self.hidden_layer1.weights = (
            self.hidden_layer1.weights - self.learning_rate * dl_dw
        )

        self.hidden_layer1.bias = self.hidden_layer1.bias - self.learning_rate * dl_db

        # for input layer 784 -> 64 -> relu -> a
        dl_dz = dl_da * derivative_ReLu(self.input_layer.a)
        dl_dw = train_x.T @ dl_dz
        dl_db = dl_dz.sum(axis=0)

        self.input_layer.weights = self.input_layer.weights - self.learning_rate * dl_dw
        self.input_layer.bias = self.input_layer.bias - self.learning_rate * dl_db


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


def derivative_ReLu(x):
    return (x >= 0).astype(float)


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def MSE_loss_fn(y_preds, y_actual, num_classes=10):
    # loss = np.mean((y_preds.flatten() - y_actual.flatten()) ** 2)

    loss = np.sum((y_preds - y_actual) ** 2, axis=1) / num_classes
    return np.mean(loss)


def Cross_Entropy_loss(y_preds, y_actual):
    loss = -np.mean(
        np.sum(y_actual * np.log(y_preds + 1e-15), axis=1)
    )  # 1e-15 as i was getting log(0) error

    return loss


def modify_y(y, num_classes):
    new_y = np.zeros((y.size, num_classes))
    for i in range(y.size):
        new_y[i][y[i]] = 1
    return new_y


def calculate_accuracy(y_preds, y_actual):
    labels = y_preds.argmax(axis=1)
    tlabels = y_actual.argmax(axis=1)

    return np.mean((labels == tlabels).astype(float))


def plot_random(x, y, model, rows=5, columns=5):
    fig, axes = plt.subplots(rows, columns, figsize=(6, 6))
    r = np.random.choice(x.shape[0], rows * columns, replace=False)
    selected = x[r]
    ty = y[r]
    y_preds = model.forward(selected)
    y_preds = y_preds.argmax(axis=1)
    ty = ty.argmax(axis=1)

    for i, ax in enumerate(axes.flat):
        img = selected[i].reshape(28, 28)
        ax.imshow(img, cmap="gray")
        ax.text(
            10,
            0,
            f"{y_preds[i]}, {ty[i]}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="green" if y_preds[i] == ty[i] else "red",
        )
        ax.axis("off")

    plt.show()


model1 = NeuralNet(
    learning_rate=0.0001, activation_fn=ReLu
)  # the weights explode on lr=0.001?? #93% accuracy on test data

model = NeuralNet(learning_rate=0.0001, activation_fn=ReLu)
y_preds = model.forward(train_x[0:20])

train_y = modify_y(train_y, 10)
validation_y_2 = modify_y(validation_y, 10)


# training batch wise
epochs = 100
batch_size = 32

batches_x = [
    train_x[i : i + batch_size] for i in range(0, train_x.shape[0], batch_size)
]
batches_y = [
    train_y[i : i + batch_size] for i in range(0, train_y.shape[0], batch_size)
]

batches_x.pop()
batches_y.pop()

for epoch in range(epochs):
    loss = 0
    acc = 0
    for i, batch in enumerate(batches_x):
        y_preds = model.forward(batch)
        model.backpropogation(batches_y[i], y_preds, batch)
        loss += Cross_Entropy_loss(y_preds, batches_y[i])
        acc += calculate_accuracy(y_preds, batches_y[i])
    loss = loss / len(batches_x)
    acc = acc / len(batches_x)
    y_preds2 = model.forward(validation_x)
    acc2 = calculate_accuracy(y_preds2, validation_y_2)
    print(
        f"Epoch: {epoch} | Loss: {loss:.4f} | Train acc: {acc * 100:.2f}| Test acc: {acc2 * 100:.2f}"
    )


# y_preds = model.forward(train_x)
# model.backpropogation(train_y, y_preds, train_x)
# loss = Cross_Entropy_loss(y_preds, train_y)
# acc = calculate_accuracy(y_preds, train_y)
# y_preds2 = model.forward(validation_x)
# acc2 = calculate_accuracy(y_preds2, validation_y_2)
# print(f"Loss: {loss} | Train acc: {acc} | Test acc: {acc2}")


# fig, axes = plt.subplots(5, 5, figsize=(6,6))
# r = np.random.choice(train_x.shape[0], 25, replace=False)
# selected = train_x[r]
# ty = train_y[r]
# y_preds = model.forward(selected)
# y_preds = y_preds.argmax(axis=1)
# ty =ty.argmax(axis=1)


# for i, ax in enumerate(axes.flat):
#     img = selected[i].reshape(28, 28)
#     ax.imshow(img, cmap='gray')
#     ax.text(10, 0, f"{y_preds[i]}, {ty[i]}",ha='center', va='bottom', fontsize=10, color='green' if y_preds[i] == ty[i] else 'red')
#     ax.axis('off')

# plt.show()


plot_random(x=validation_x, y=validation_y_2, model=model, rows=8, columns=8)
