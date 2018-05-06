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
    x, y = Variable(torch.FloatTensor(x)), Variable(torch.Tensor(y))
    if use_cuda:
        x, y = x.cuda(), y.cuda()
    
    optimizer.zero_grad()
    outputs = net(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    _, predicted = torch.max(outputs.data, 1)
    _, targets = torch.max(y.data, 1)
    correct = predicted.eq(targets).sum()
    return loss.item(), correct

def test(net, test_data, use_cuda):
    net.eval()
    criterion = nn.MSELoss()
    x, y = test_data
    
    x, y = Variable(torch.FloatTensor(x)), Variable(torch.Tensor(y))
    if use_cuda:
        x, y = x.cuda(), y.cuda()

    outputs = net(x)
    loss = criterion(outputs, y)

    total = len(x)
    _, predicted = torch.max(outputs.data, 1)
    _, targets = torch.max(y.data, 1)
    correct = predicted.eq(targets).sum()

    # return 'Test loss: %.3f | Acc: %.3f%% (%d/%d) \n' % (
    #     loss.item(), 100.0 * correct / total, correct, total)
    return loss.item(), correct


























