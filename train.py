import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

def train(net, train_data, use_cuda):
    net.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters())

    train_loss = 0
    total, correct = 0, 0

    x, y = train_data

    if use_cuda:
        x, y = x.cuda(), y.cuda()

    optimizer.zero_grad()
    outputs = net(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    _, predicted = torch.max(outputs.data, 1)
    correct = predicted.eq(y.data).cpu().sum()
    return loss[0], correct

def test(net, test_data, use_cuda):
    net.eval()
    criterion = nn.MSELoss()
    x, y = test_data

    if use_cuda:
        x, y = x.cuda(), y.cuda()

    x, y = Variable(torch.Tensor(x)), Variable(torch.Tensor(y))
    outputs = net(x)
    loss = criterion(outputs, y)

    total = len(x)
    _, predicted = torch.max(outputs.data, 1)
    correct = predicted.eq(y.data).cpu().sum()

    print 'Test loss: %.3f | Acc: %.3f%% (%d/%d)' % (
        loss.data[0], 100.0 * correct / total, correct, total)



























