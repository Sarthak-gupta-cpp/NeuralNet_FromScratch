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
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.z = 0
        # self.neurons = np.array([Neuron(input_shape) for i in range(output_shape)])  (making a list of neurons was not efficient and i was not able to include batches using this)
        self.weights = np.random.randn(input_shape, output_shape) * 0.01
        self.bias = np.random.randn(output_shape) * 0.01
        self.input = 0

    def forward(self, x):
        # return self.activation_fn(np.array([i.forward(x) for i in self.neurons]))
        self.z = np.matmul(x, self.weights) + self.bias
        self.input = x
        return self.z

    def backward(self, dl_dout, lr):
        dl_dx = dl_dout @ self.weights.T
        dl_dw = self.input.T @ dl_dout
        dl_db = np.sum(dl_dout, axis=0)

        self.weights -= dl_dw * lr
        self.bias -= dl_db * lr


class Max_Pooling_Layer:
    def __init__(self, kernel=2, stride=2):
        self.k = kernel
        self.s = stride

    def forward(self, x):
        # x is (10, 10, 27, 27)
        out_shape = int((x.shape[2] - self.k) / self.s + 1)
        out = np.zeros((x.shape[0], x.shape[1], out_shape, out_shape))

        ai = 0
        aj = 0
        for b in range(0, x.shape[0]):
            for f in range(0, x.shape[1]):
                for i in range(0, x.shape[2] - 1, self.s):
                    for j in range(0, x.shape[3] - 1, self.s):
                        sel = x[b, f, i : i + self.k, j : j + self.k]
                        m = np.max(sel)
                        out[b][f][ai][aj] = m
                        aj += 1
                    ai += 1
                    aj = 0
                ai = 0

        print(out.shape)

        # for j in range(out.shape[0]):
        #     fig, axes = plt.subplots(5, 2, figsize=(6, 6))
        #     for i, ax in enumerate(axes.flat):
        #         img = out[j][i]
        #         ax.imshow(img, cmap="seismic")
        #         ax.axis("off")

        fig, axes = plt.subplots(5, 2, figsize=(6, 6))
        for i, ax in enumerate(axes.flat):
            img = out[0][i]
            ax.imshow(img, cmap="seismic")
            ax.axis("off")

        return out


class Convulution_Layer:
    def __init__(
        self,
        kernel=2,
        stride=1,
        padding=0,
        input_shape=1,
        output_shape=10,
    ):
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weights = np.random.randn(
            self.output_shape, self.input_shape * kernel * kernel
        )  # (10, 1*2*2)
        self.biases = np.random.randn(output_shape)

    def im2col(self, x):
        b, c, h, w = x.shape  # (h=w for mnist)
        cols = []
        for i in range(0, h - self.kernel + 1, self.stride):
            for j in range(0, w - self.kernel + 1, self.stride):
                patch = x[
                    :, :, i : i + self.kernel, j : j + self.kernel
                ]  # (B, c, k, k)
                cols.append(patch.reshape(b, -1))  # (b, c*k*k)

        col = np.stack(cols, axis=1)
        print(col.shape)  # (10, 729, 4)
        return col

    def forward(self, x):
        # x is (10, 1, 28, 28)

        b, c, h, w = x.shape

        out_h = (h + 2 * self.padding - self.kernel) // self.stride + 1
        out_w = (w + 2 * self.padding - self.kernel) // self.stride + 1

        x_col = self.im2col(x)  # (10, 729, 4)

        # (b, n, c)
        out = x_col @ self.weights.T  # (10, 729, 10)

        out = out.transpose(0, 2, 1)  # (b, c, n) (10, 10, 729)
        out = out.reshape(b, self.output_shape, out_h, out_w)  # (10, 10, 27, 27)

        print(out.shape)

        # for j in range(out.shape[0]):
        #     fig, axes = plt.subplots(5, 2, figsize=(6, 6))
        #     for i, ax in enumerate(axes.flat):
        #         img = out[j][i]
        #         ax.imshow(img, cmap="seismic")
        #         ax.axis("off")

        fig, axes = plt.subplots(5, 2, figsize=(6, 6))
        for i, ax in enumerate(axes.flat):
            img = out[0][i]
            ax.imshow(img, cmap="seismic")
            ax.axis("off")

        return out


Layert = Convulution_Layer()
out = Layert.forward(train_x[10:20].reshape(10, 1, 28, 28))


out = Layert.forward(train_x[0:10].reshape(10, 1, 28, 28))

Layerth = Convulution_Layer(input_shape=10)
out = Layerth.forward(out)

LayerM = Max_Pooling_Layer()
out = LayerM.forward(out)


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
            activation_fn=activation_fn.fn,
        )
        self.hidden_layer1 = Linear_Layer(
            input_shape=hidden_layer,
            output_shape=hidden_layer,
            activation_fn=activation_fn.fn,
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
        dl_dz = dl_da * self.act.der(self.hidden_layer1.a)  # (32x64)
        dl_dw = self.input_layer.a.T @ dl_dz
        dl_db = dl_dz.sum(axis=0)
        dl_da = dl_dz @ self.hidden_layer1.weights.T

        self.hidden_layer1.weights = (
            self.hidden_layer1.weights - self.learning_rate * dl_dw
        )

        self.hidden_layer1.bias = self.hidden_layer1.bias - self.learning_rate * dl_db

        # for input layer 784 -> 64 -> relu -> a
        dl_dz = dl_da * self.act.der(self.input_layer.a)
        dl_dw = train_x.T @ dl_dz
        dl_db = dl_dz.sum(axis=0)

        self.input_layer.weights = self.input_layer.weights - self.learning_rate * dl_dw
        self.input_layer.bias = self.input_layer.bias - self.learning_rate * dl_db


class CNNNeuralNet:
    def __init__(
        self,
        input_shape=1,
        output_shape=10,
        filters=32,
        learning_rate=0.001,
        activation_fn=lambda x: x,
        kernel=2,
        padding=0,
        stride=1,
        mkernel=2,
        mstride=2,
        image_size=28,
    ):
        self.act = activation_fn
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learning_rate = learning_rate
        self.Layer1 = Convulution_Layer(
            kernel=kernel,
            padding=0,
            stride=stride,
            output_shape=filters,
            input_shape=input_shape,
        )
        self.Layer2 = Max_Pooling_Layer(kernel=mkernel, stride=mstride)
        self.Layer3 = Convulution_Layer(
            kernel=kernel,
            padding=0,
            stride=stride,
            output_shape=filters,
            input_shape=filters,
        )
        self.Layer4 = Max_Pooling_Layer(kernel=mkernel, stride=mstride)

        shape = image_size
        shape = (shape + 2 * padding - kernel) // stride + 1  # layer1
        shape = (shape - mkernel) // mstride + 1  # layer2
        shape = (shape + 2 * padding - kernel) // stride + 1  # layer3
        shape = (shape - mkernel) // mstride + 1  # layer4

        self.Layer5 = Linear_Layer(
            input_shape=shape * shape * filters,
            output_shape=32,
            activation_fn=activation_fn,
        )
        self.Layer6 = Linear_Layer(
            input_shape=32, output_shape=output_shape, activation_fn=activation_fn
        )

    def forward(self, x):
        y = self.Layer1.forward(x)  # conv
        y = self.act.fn(y)  # ReLu
        y = self.Layer2.forward(y)  # maxpool
        y = self.Layer3.forward(y)  # conv
        y = self.act.fn(y)  # ReLu
        y = self.Layer4.forward(y)  # maxpool
        y = y.reshape(y.shape[0], -1)
        y = self.Layer5.forward(y)  # Linear
        y = self.act.fn(y)  # ReLu
        y = self.Layer6.forward(y)  # Linear
        y = self.act.fn(y)  # ReLu

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
        dl_dz = dl_da * self.act.der(self.hidden_layer1.a)  # (32x64)
        dl_dw = self.input_layer.a.T @ dl_dz
        dl_db = dl_dz.sum(axis=0)
        dl_da = dl_dz @ self.hidden_layer1.weights.T

        self.hidden_layer1.weights = (
            self.hidden_layer1.weights - self.learning_rate * dl_dw
        )

        self.hidden_layer1.bias = self.hidden_layer1.bias - self.learning_rate * dl_db

        # for input layer 784 -> 64 -> relu -> a
        dl_dz = dl_da * self.act.der(self.input_layer.a)
        dl_dw = train_x.T @ dl_dz
        dl_db = dl_dz.sum(axis=0)
        dl_da = dl_dz @ self.output_shape

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


class ReLu:
    def fn(self, x):
        return np.maximum(x, 0)

    def der(self, x):
        return (x >= 0).astype(float)


class Leaky_ReLu:
    def __init__(self, alpha):
        self.alpha = alpha

    def fn(self, x):
        return np.maximum(x, self.alpha * x)

    def der(self, x):
        arr = x >= 0
        return (arr - 1) * (1 - self.alpha) + 1


# def derivative


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


def plot_image(x):
    plt.imshow(x, cmap="gray")
    plt.show()


train_y = modify_y(train_y, 10)
validation_y_2 = modify_y(validation_y, 10)

model1 = NeuralNet(
    learning_rate=0.0001, activation_fn=ReLu
)  # the weights explode on lr=0.001?? #93% accuracy on test data

model2 = NeuralNet(
    learning_rate=0.0001, activation_fn=Leaky_ReLu(0.01)
)  # 95% acc with leaky ReLu(0.01)

model = CNNNeuralNet(learning_rate=0.0001, activation_fn=ReLu())
out = model.forward(train_x[0:10].reshape(10, 1, 28, 28))


y_preds = model.forward(train_x[0:20])


# training batch wise
epochs = 10
batch_size = 64

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
