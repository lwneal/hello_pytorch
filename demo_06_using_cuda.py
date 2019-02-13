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

def cross_entropy(y, y_hat):
    eps = .000001
    return -(y * torch.log(y_hat + eps)).mean()

class ClassifierNet(torch.nn.Module):
    def __init__(self, num_hidden=128):
        super().__init__()
        self.fc1 = torch.nn.Linear(num_input_features, num_hidden)
        self.fc2 = torch.nn.Linear(num_hidden, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.softmax(x, dim=1)
        return x

x = torch.Tensor(input_data)
y = torch.Tensor(integer_to_categorical(input_labels))
model = ClassifierNet()
optimizer = torch.optim.SGD(model.parameters(), lr=.03)

x = x.cuda()
y = y.cuda()
model = model.cuda()

for i in range(3000):
    y_hat = model(x)
    loss = cross_entropy(y, y_hat)

    # Compute the gradient of the loss wrt. W and b
    loss.backward()

    # Perform one step of gradient descent
    optimizer.step()
    optimizer.zero_grad()
    print('Cross entropy loss is {:.3f}'.format(loss))

predicted_labels = y_hat.argmax(dim=1).cpu().numpy()
num_correct = (predicted_labels == input_labels).sum()
print('Classified {}/{} correct, loss {:.3f}'.format(
    num_correct, num_examples, loss))

