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
        self.fc2 = nn.Linear(32,25)
        self.fc3 = nn.Linear(25,20)
        self.fc4 = nn.Linear(20,15)
        self.fc5 = nn.Linear(15,10)
        self.fc6 = nn.Linear(10,5)
        self.fc7 = nn.Linear(5,2)

    def forward(self,x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        x = F.sigmoid(self.fc5(x))
        x = F.sigmoid(self.fc6(x))
        x = self.fc7(x)
        return x

net = Net()
net.cuda()

#use cross entropy loss
criterion = nn.CrossEntropyLoss()
#use SGD
optimizer = optim.SGD(net.parameters(), lr=0.0000000000001, momentum=0.9)

data_file = open('data/train')
data_read = data_file.readlines()

#set up batch size
BATCH_SIZE=100

for epoch in range(100):    # loop over the dataset multiple times

    running_loss = 0.0
    for i in range(100):
        inputs = torch.Tensor(BATCH_SIZE,32)
        labels = torch.LongTensor(BATCH_SIZE)
        for j in range(int(BATCH_SIZE)):
            data_array = data_read[random.randint(0,3999)].split()
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
           # data_array_2 = data_read[500 + int(BATCH_SIZE/2) * i + j].split()
            # get the sequential number
            #seq = int(data_array[0][1:len(data_array[0])-1])

            #get the label
           # if data_array_2[1] == '1':
                #labels[int(BATCH_SIZE/2)+j] = 1
            #else:
                #labels[int(BATCH_SIZE/2)+j] = 0
                #Gets the data
                #for k in range(32):
                    #inputs[j][k] = float(data_array_2[2+k])


        # wrap them in Variable
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs.cuda(), labels.cuda())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 50 == 49:        # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss))
            running_loss = 0.0

print('Finished Training')

#calculate accuracy
correct = 0
total = 0
test_inputs = torch.rand(1,32)
test_labels = torch.LongTensor(1)
#for data in testloader:
test_file = open('data/test')
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

print('Accuracy of the network: %d %%' % (
    100 * correct / total))
