import torch
import numpy as np
import random

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

#classes = ('yes', 'no')

#setup fully connected neural networks with 3 hidden layers
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32,32)
        self.fc3 = nn.Linear(32,10)
        self.fc4 = nn.Linear(10,2)
        #self.fc5 = nn.Linear(15,10)
        #self.fc6 = nn.Linear(10,5)
        #self.fc7 = nn.Linear(5,2)

    def forward(self,x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        #x = F.sigmoid(self.fc4(x))
        #x = F.sigmoid(self.fc5(x))
        #x = F.sigmoid(self.fc6(x))
        x = self.fc4(x)
        return x

net = Net()

#use loss function as cross entropy loss
criterion = nn.CrossEntropyLoss()

#select training algorithm
optimizer = optim.Adam(net.parameters(), lr=0.00004)

#indicate training & testing set size
training_size = 6000
testing_size = 200

data_file = open('data/train2')
data_read = data_file.readlines()

#set up batch size
BATCH_SIZE=60

for epoch in range(500):    # loop over the dataset multiple times

    running_loss = 0.0
    for i in range(100):
        inputs = torch.Tensor(BATCH_SIZE,32)
        labels = torch.LongTensor(BATCH_SIZE)
        for j in range(int(BATCH_SIZE)):
            data_array = data_read[random.randint(0,training_size-1)].split()
            # get the sequential number
            #seq = int(data_array[0][1:len(data_array[0])-1])

            #get the label
            if data_array[1] == '1':
                labels[j] = 1
            else:
                labels[j] = 0
                #Gets the data
                for k in range(32):
                    inputs[j][k] = float(data_array[2+k])

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 100 == 99:       # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss))
            running_loss = 0.0

print('Finished Training')

#calculate accuracy
correct = 0
total = 0
test_inputs = torch.rand(1,32)
test_labels = torch.LongTensor(1)

#load test data from file
test_file = open('data/train')
test_read = test_file.readlines()

for test_line in test_read:
    test_array = test_line.split()
    if test_array[1] == '1':
        test_labels=1
    else:
        test_labels=0
    for k in range(32):
        test_inputs[0][k] = float(test_array[2+k])
    #images, labels = data
    #print(test_inputs)
    test_outputs = net(Variable(test_inputs))
    #print(test_outputs)
    _, predicted = torch.max(test_outputs.data, 1)
    total += 1
    correct += (predicted == test_labels).sum()
    if(total==100): print('Accuracy on the first half: %d %%' % (
    100 * correct / total))

print('Accuracy on test: %d %%' % (
    100 * correct / total))

correct = 0
total = 0
train_inputs = torch.rand(1,32)
train_labels = torch.LongTensor(1)

train_read=data_read[0:100]
train_read.extend(data_read[2000:2000+100])

for train_line in train_read:
    train_array = train_line.split()
    if train_array[1] == '1':
        train_labels=1
    else:
        train_labels=0
    for k in range(32):
        train_inputs[0][k] = float(train_array[2+k])
    #images, labels = data
    #print(train_inputs)
    train_outputs = net(Variable(train_inputs))
    #print(train_outputs)
    _, predicted = torch.max(train_outputs.data, 1)
    total += 1
    correct += (predicted == train_labels).sum()
    if(total==100): print('Accuracy on the first half: %d %%' % (
    100 * correct / total))

print('Accuracy on train: %d %%' % (
    100 * correct / total))
