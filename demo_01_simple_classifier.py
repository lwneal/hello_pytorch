# CS536: Deep Learning
# Introduction to PyTorch
import numpy as np
from sklearn import datasets

input_data, input_labels = datasets.load_iris(return_X_y=True)
num_examples, num_input_features = input_data.shape
num_classes = input_labels.max() + 1

def integer_to_categorical(x, N=None):
    if N is None:
        N = x.max() + 1
    return np.eye(N)[x]

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def matmul(W, x):
    return np.einsum('bi,ji->bj', x, W)

def cross_entropy(y, y_hat):
    eps = .000001
    return -(y * np.log(y_hat + eps)).mean()


x = np.array(input_data)
y = np.array(integer_to_categorical(input_labels))
W = np.random.normal(size=(num_classes, num_input_features))
b = np.random.normal(size=(num_classes))

for i in range(100):
    y_hat = softmax(matmul(W, x) + b)
    loss = cross_entropy(y, y_hat)

    # TODO: Write backpropagation code here to update parameters
    W_grad = np.zeros(W.shape)
    b_grad = np.zeros(b.shape)
    # TODO: compute W_grad and b_grad

    # Perform one step of gradient descent
    W = W - .01 * W_grad
    b = b - .01 * b_grad
    print('Cross entropy loss is {:.3f}'.format(loss))

predicted_labels = y_hat.argmax(axis=1)
num_correct = (predicted_labels == input_labels).sum()
print('Classified {}/{} correct, loss {:.3f}'.format(
    num_correct, num_examples, loss))

