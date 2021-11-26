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
import os
import random
import cv2

from nn import *

####################################### 6.1.1. #######################################
print('6.1.1. Torch')

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
    print('Skipped training for 6.1.1.')

####################################### 6.1.2. #######################################
print('6.1.2. CNN')
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
    print('Skipped training for 6.1.2.')

####################################### 6.1.3. #######################################
print('6.1.3. CIFAR')
# Batch size
batch_size = 50

# Get CIFAR10 dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

class CIFARNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
cifar_net = CIFARNet()

# Trainer
optimizer = optim.SGD(cifar_net.parameters(), lr=learning_rate, momentum=momentum)

# Training
avg_loss_matrix = []
avg_acc_matrix = []
if False:
    for itr in range(max_iters):
        avg_loss = 0
        avg_acc = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = cifar_net(inputs)
            # Accuracy
            _, predictions = torch.max(outputs, dim=1)
            avg_acc += torch.count_nonzero(predictions == labels).item()
            # Loss
            loss = criterion(outputs, labels)
            avg_loss += loss.item()
            #  backward + optimize
            loss.backward()
            optimizer.step()
        # Average loss and accuracy over training set
        avg_acc /= (i+1)*batch_size
        avg_loss /= (i+1)*batch_size
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
    print('Skipped training for 6.1.3.')

####################################### 6.1.4. #######################################
print('6.1.4. HW1 Revisited')

# Get files
hw1_data_path = '../../HW1/data'
train_files_txt = 'train_files.txt'
train_labels_txt = 'train_labels.txt'
train_files_dir = os.path.join(hw1_data_path, train_files_txt)
train_labels_dir = os.path.join(hw1_data_path, train_labels_txt)
train_files_list = []
with open(train_files_dir, 'r') as train_files_obj:
    for line in train_files_obj:
        train_files_list.append(os.path.join(hw1_data_path, line).strip())
train_labels_list = []
with open(train_labels_dir, 'r') as train_labels_obj:
    for line in train_labels_obj:
        train_labels_list.append(int(line.strip()))
# Shuffle training
temp = list(zip(train_files_list, train_labels_list))
random.shuffle(temp)
train_x_dir, train_y = zip(*temp)

# Net
class BOWNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 47 * 47, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 8)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
bownet = BOWNet()

# Trainer
optimizer = optim.SGD(bownet.parameters(), lr=learning_rate/(5*1177), momentum=momentum)

# Batch size
batch_size = 1

# Epochs
max_iters = 15

# Training
avg_loss_matrix = []
avg_acc_matrix = []
img_size = (200, 200)
if True:
    for itr in range(max_iters):
        avg_loss = 0
        avg_acc = 0
        for i, (x_dir, y) in enumerate(zip(train_x_dir, train_y)):
            x_org = cv2.resize(cv2.imread(x_dir), img_size)
            x = np.zeros((3,x_org.shape[0], x_org.shape[1]))
            if(len(x.shape)<3):
                # x is grayscale
                x[0] = x_org
                x[1] = x_org
                x[2] = x_org
            else:
                x[0] = x_org[:,:,0]
                x[1] = x_org[:,:,1]
                x[2] = x_org[:,:,2]
            # Batch size of 1
            inputs = torch.from_numpy(np.expand_dims(x,axis=0)).type(torch.float32)
            labels = torch.tensor([y])
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = bownet(inputs)
            # Accuracy
            _, predictions = torch.max(outputs, dim=1)
            avg_acc += torch.count_nonzero(predictions == labels).item()
            # Loss
            loss = criterion(outputs, labels)
            avg_loss += loss.item()
            #  backward + optimize
            loss.backward()
            optimizer.step()
        # Average loss and accuracy over training set
        avg_acc /= (i+1)*batch_size
        avg_loss /= (i+1)*batch_size
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
    print('Skipped training for 6.1.4.')
pass

