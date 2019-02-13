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


x = torch.Tensor(input_data)
y = torch.Tensor(integer_to_categorical(input_labels))
W = torch.Tensor(num_classes, num_input_features).normal_(0, .1)
b = torch.Tensor(num_classes).normal_()
W.requires_grad = True
b.requires_grad = True

for i in range(2000):
    y_hat = softmax(matmul(W, x) + b)
    loss = cross_entropy(y, y_hat)

    # Compute the gradient of the loss wrt. W and b
    loss.backward()

    # Perform one step of gradient descent
    W.data -= .03 * W.grad
    b.data -= .03 * b.grad
    W.grad.data.zero_()
    b.grad.data.zero_()
    print('Cross entropy loss is {:.3f}'.format(loss))

predicted_labels = y_hat.argmax(dim=1).numpy()
num_correct = (predicted_labels == input_labels).sum()
print('Classified {}/{} correct, loss {:.3f}'.format(
    num_correct, num_examples, loss))

