# CS536: Deep Learning
# Introduction to PyTorch
import torch
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
    return torch.softmax(x, dim=1)

def matmul(W, x):
    return torch.einsum('ij,bj->bi', (W, x))

def cross_entropy(y, y_hat):
    eps = .000001
    return -(y * torch.log(y_hat + eps)).mean()

num_hidden = 128
x = torch.Tensor(input_data)
y = torch.Tensor(integer_to_categorical(input_labels))
W1 = torch.Tensor(num_hidden, num_input_features).normal_(0, .01)
b1 = torch.Tensor(num_hidden).normal_(0, .1)
W2 = torch.Tensor(num_classes, num_hidden).normal_(0, .1)
b2 = torch.Tensor(num_classes).normal_(0, .1)
W1.requires_grad = True
b1.requires_grad = True
W2.requires_grad = True
b2.requires_grad = True

for i in range(3000):
    h = torch.sigmoid(matmul(W1, x) + b1)
    y_hat = softmax(matmul(W2, h) + b2)
    loss = cross_entropy(y, y_hat)

    # Compute the gradient of the loss wrt. W and b
    loss.backward()

    # Perform one step of gradient descent
    W1.data -= .03 * W1.grad
    b1.data -= .03 * b1.grad
    W2.data -= .03 * W2.grad
    b2.data -= .03 * b2.grad
    W1.grad.data.zero_()
    b1.grad.data.zero_()
    W2.grad.data.zero_()
    b2.grad.data.zero_()
    print('Cross entropy loss is {:.3f}'.format(loss))

predicted_labels = y_hat.argmax(dim=1).numpy()
num_correct = (predicted_labels == input_labels).sum()
print('Classified {}/{} correct, loss {:.3f}'.format(
    num_correct, num_examples, loss))

