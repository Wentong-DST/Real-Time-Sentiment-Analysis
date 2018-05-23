import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

def train(net, train_data, args, embed_flag=0, fine_tune = False):
    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())

    train_loss = 0
    total, correct = 0, 0

    x, y = train_data
    if embed_flag:
        x, y = Variable(torch.LongTensor(x), requires_grad=fine_tune), Variable(torch.LongTensor(y))
    else:
        x, y = Variable(torch.FloatTensor(x), requires_grad=fine_tune), Variable(torch.LongTensor(y))
    if args.use_cuda:
        x, y = x.cuda(), y.cuda()
    
    optimizer.zero_grad()
    outputs = net(x, embed_flag)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    _, predicted = torch.max(outputs.data, 1)
#    _, targets = torch.max(y.data, 1)
    targets = y.data
    correct = predicted.eq(targets).sum()
    if fine_tune:
        if x.grad is not None:
            x.data -= args.lr * x.grad.data
        return loss.item(), correct, np.array(x.data)
    return loss.item(), correct

def test(net, test_data, use_cuda, embed_flag=0):
    net.eval()
    criterion = nn.CrossEntropyLoss()
    x, y = test_data
    
    if embed_flag:
        x, y = Variable(torch.LongTensor(x)), Variable(torch.LongTensor(y))
    else:
        x, y = Variable(torch.FloatTensor(x)), Variable(torch.LongTensor(y))
    if use_cuda:
        x, y = x.cuda(), y.cuda()

    outputs = net(x, embed_flag)
    loss = criterion(outputs, y)

    total = len(x)
    _, predicted = torch.max(outputs.data, 1)
    #_, targets = torch.max(y.data, 1)
    targets = y.data
    correct = predicted.eq(targets).sum()

    # return 'Test loss: %.3f | Acc: %.3f%% (%d/%d) \n' % (
    #     loss.item(), 100.0 * correct / total, correct, total)
    return loss.item(), correct


























