import argparse
import torch
import json
import optmizor
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from model import NonConvex
from tqdm import tqdm
from torch import nn
import os
from utils.ploter import AverageMeter, RecordWriter

MODEL_DICT = {
    'Func1': NonConvex.Function1
}


def main(args):
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    print('device {}'.format(device))

    model = NonConvex.Function1().to(device)

    print(model)


    if args.optimizer == 'SGD':
        writer = RecordWriter('NonConvex_{}_{}'.format(args.optimizer, args.lr))
        optimizer = optmizor.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'SGD-M':
        optimizer = optmizor.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'FGD-K':
        writer = RecordWriter('NonConvex_{}_{}'.format(args.optimizer, args.lr))
        optimizer = optmizor.KGD(model.parameters(), lr=args.lr, device=device)
    elif args.optimizer == 'ARMAGD':
        print(args.memory)
        writer = RecordWriter('NonConvex_{}_{}_{}'.format(args.optimizer, args.lr, args.memory))
        optimizer = optmizor.ARMAGD(model.parameters(), lr=args.lr, memory=args.memory, device=device)
    elif args.optimizer == 'MASGD':
        print(args.optimizer, args.memory)
        writer = RecordWriter('NonConvex_{}_{}_{}'.format(args.optimizer, args.lr, args.memory))
        optimizer = optmizor.MASGD(model.parameters(), lr=args.lr, memory=args.memory, device=device)
    elif args.optimizer == 'FGD-W':
        print(args.optimizer)
        writer = RecordWriter('NonConvex_{}_{}'.format(args.optimizer, args.lr))
        optimizer = optmizor.WTGD(model.parameters(), lr=args.lr, device=device)
    elif args.optimizer == 'Adam':
        print(args.optimizer)
        writer = RecordWriter('NonConvex_{}_{}'.format(args.optimizer, args.lr))
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        print('Optimizer not defined')
        exit()


    writer.update('x', [model.x.data.cpu().numpy().tolist()])
    writer.update('y', [model.y.data.cpu().numpy().tolist()])
    for epoch in range(1, args.epochs + 1):
        output = model()
        optimizer.zero_grad()
        print('[{}/{}] Output:{}'.format(epoch, args.epochs + 1, output))
        output.backward()
        optimizer.step()
        writer.update('output', [output.data.cpu().numpy().tolist()])
        writer.update('x', [model.x.data.cpu().numpy().tolist()])
        writer.update('x_grad', [model.x.grad.cpu().numpy().tolist()])
        writer.update('y', [model.y.data.cpu().numpy().tolist()])
        writer.update('y_grad', [model.y.grad.cpu().numpy().tolist()])


    writer.write()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--memory', nargs="+", type=float, default=[0.1, 0.8])
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.03, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='MOM',
                        help='momentum (default: 0.9)')
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='disables CUDA training')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--cuda', type=str, default='0')

    args = parser.parse_args()
    main(args)
