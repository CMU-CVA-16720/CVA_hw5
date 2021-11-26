import torch
from torch.utils.data.dataset import TensorDataset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import scipy.io

from nn import *

####################################### 6.1.1. #######################################

# Batch size
batch_size = 50

# Epochs
max_iters = 50
# max_iters = 1

# Learning rate & momentum
learning_rate = 1e-2
momentum = 0.9

# Get data
train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')
train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']
# Convert to tensors - training
train_x_tensor, train_y_tensor = torch.from_numpy(train_x).type(torch.float32), torch.from_numpy(train_y)
trainset = TensorDataset(train_x_tensor, train_y_tensor)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=2)
# Convert to tensors - validation
valid_x_tensor, valid_y_tensor = torch.from_numpy(valid_x).type(torch.float32), torch.from_numpy(valid_y)
validset = TensorDataset(valid_x_tensor, valid_y_tensor)
valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size,shuffle=True, num_workers=2)
# Convert to tensors - testing
test_x_tensor, test_y_tensor = torch.from_numpy(test_x).type(torch.float32), torch.from_numpy(test_y)
testset = TensorDataset(test_x_tensor, test_y_tensor)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=True, num_workers=2)

# Network size
in_size = train_x.shape[1]
hidden_size = 64
out_size = train_y.shape[1]

# Shuffle
batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

# Define neural net
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x
net = Net()

# Loss & trainer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

# Training
avg_loss_matrix = []
avg_acc_matrix = []
if False:
    for itr in range(max_iters):
        avg_loss = 0
        avg_acc = 0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            _, answers = torch.max(labels, dim=1)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = net(inputs)
            # Accuracy
            _, predictions = torch.max(outputs, dim=1)
            avg_acc += torch.count_nonzero(predictions == answers).item()
            # Loss
            loss = criterion(outputs, answers)
            avg_loss += loss.item()
            #  backward + optimize
            loss.backward()
            optimizer.step()
        # Average loss and accuracy over training set
        avg_acc /= train_x.shape[0]
        avg_loss /= train_x.shape[0]
        avg_acc_matrix.append(avg_acc)
        avg_loss_matrix.append(avg_loss)
        print('{} - Acc/loss = {} / {}'.format(itr, avg_acc, avg_loss))
    print('Finished Training')
    # Graph accuracy
    ax = plt.axes()
    ax.plot(np.arange(0,max_iters), avg_acc_matrix, color='red') # training acc
    plt.xlim(0, max_iters)
    plt.ylim(0, 1)
    plt.title("Training Acc vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Acc (%)")
    plt.show()
    # Graph loss
    ax = plt.axes()
    ax.plot(np.arange(0,max_iters), avg_loss_matrix, color='red') # training loss
    plt.xlim(0, max_iters)
    plt.title("Training Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
else:
    print('Skipped training')

####################################### 6.1.2. #######################################
# Using CNN shown in tutorial, modified for our use
class CNNet(nn.Module):
    def __init__(self):
        super(CNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 36)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
cnnet = CNNet()

# Trainer
optimizer = optim.SGD(cnnet.parameters(), lr=learning_rate, momentum=momentum)

# Training
avg_loss_matrix = []
avg_acc_matrix = []
if True:
    for itr in range(max_iters):
        avg_loss = 0
        avg_acc = 0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            _, answers = torch.max(labels, dim=1)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = cnnet(torch.unsqueeze(inputs.reshape((50,32,32)), dim=1))
            # Accuracy
            _, predictions = torch.max(outputs, dim=1)
            avg_acc += torch.count_nonzero(predictions == answers).item()
            # Loss
            loss = criterion(outputs, answers)
            avg_loss += loss.item()
            #  backward + optimize
            loss.backward()
            optimizer.step()
        # Average loss and accuracy over training set
        avg_acc /= train_x.shape[0]
        avg_loss /= train_x.shape[0]
        avg_acc_matrix.append(avg_acc)
        avg_loss_matrix.append(avg_loss)
        print('{} - Acc/loss = {} / {}'.format(itr, avg_acc, avg_loss))
    print('Finished Training')
    # Graph accuracy
    ax = plt.axes()
    ax.plot(np.arange(0,max_iters), avg_acc_matrix, color='red') # training acc
    plt.xlim(0, max_iters)
    plt.ylim(0, 1)
    plt.title("Training Acc vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Acc (%)")
    plt.show()
    # Graph loss
    ax = plt.axes()
    ax.plot(np.arange(0,max_iters), avg_loss_matrix, color='red') # training loss
    plt.xlim(0, max_iters)
    plt.title("Training Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
else:
    print('Skipped training')