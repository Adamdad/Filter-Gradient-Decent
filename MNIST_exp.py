import argparse
import torch
import json
import optmizor
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from model.MNIST import Resnet
from model import MLP
from tqdm import tqdm
from torch import nn
import os
from utils.ploter import AverageMeter, RecordWriter

MODEL_DICT = {
    'resnet18':Resnet.resnet18,
    'resnet34':Resnet.resnet34,
    'resnet50': Resnet.resnet50,
    'MLP': MLP.MLP
}


def train(args, model, device, train_loader, optimizer, epoch, LOSS_FUNC):
    losses = AverageMeter('Loss', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    model.train()
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, ascii=True)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = LOSS_FUNC(output, target)
        loss.backward()
        optimizer.step()

        acc1 = accuracy(output, target, topk=(1,))[0]
        losses.update(loss.item(), data[0].size(0))
        top1.update(acc1[0].item(), data[0].size(0))
        if batch_idx % args.log_interval == 0:
            tqdm.write('Epoch [%d/%d]\tIter [%d/%d]\tAvg Loss: %.4f\tLoss: %.4f\tAvg Acc1: %.4f\tAcc1: %.4f'
                       % (epoch, args.epochs, batch_idx + 1, len(train_loader), losses.avg, loss.item(), top1.avg,
                          acc1[0]))
    return losses, top1


def test(args, model, device, test_loader, LOSS_FUNC):
    losses = AverageMeter('Loss', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    model.eval()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = LOSS_FUNC(output, target)
            acc1 = accuracy(output, target, topk=(1,))[0]
            losses.update(loss.item(), data[0].size(0))
            top1.update(acc1[0].item(), data[0].size(0))

    print('\nTest set: Average loss: {:.4f}, Accuracy: ({:.3f}%)\n'.format(
        losses.avg, top1.avg))
    return losses, top1


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def main(args):
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    if os.path.exists(args.checkpoint_path) == False:
        os.makedirs(args.checkpoint_path)

    print('device {}\t save model {}'.format(device,args.save_model))
    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(2),
                           transforms.RandomCrop(28),
                           transforms.ToTensor(),
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    NUMBER_CLS = 10
    if args.model_name == 'MLP':
        model = MODEL_DICT[args.model_name](args=args.layers).to(device)
    else:
        model = MODEL_DICT[args.model_name](num_classes=NUMBER_CLS).to(device)
    print(model)
    LOSS_FUNC = nn.CrossEntropyLoss().to(device)
    print(args.optimizer, args.batch_size, args.lr)
    if args.optimizer == 'SGD':
        writer = RecordWriter(
            'MNIST_{}_{}_{}_{}'.format(args.optimizer, args.batch_size, args.model_name, args.lr))
        optimizer = optmizor.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'SGD-M':
        writer = RecordWriter(
            'MNIST_{}_{}_{}_{}_{}'.format(args.optimizer, args.batch_size, args.model_name, args.lr, args.momentum))
        optimizer = optmizor.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'KO':
        writer = RecordWriter(
            'MNIST_{}_{}_{}_{}'.format(args.optimizer, args.batch_size, args.model_name, args.lr))
        optimizer = optmizor.KGD(model.parameters(), lr=args.lr, device=device)
    elif args.optimizer == 'ARMAGD':
        writer = RecordWriter(
            'MNIST_{}_{}_{}_{}_{}'.format(args.optimizer, args.batch_size, args.model_name, args.lr, args.memory))
        optimizer = optmizor.ARMAGD(model.parameters(), lr=args.lr, memory=args.memory, device=device)
    elif args.optimizer == 'MASGD':
        writer = RecordWriter(
            'MNIST_{}_{}_{}_{}_{}'.format(args.optimizer, args.batch_size, args.model_name, args.lr, args.memory))

        optimizer = optmizor.MASGD(model.parameters(), lr=args.lr, memory=args.memory, device=device)
    elif args.optimizer == 'WTSGD':
        writer = RecordWriter(
            'MNIST_{}_{}_{}_{}'.format(args.optimizer, args.batch_size, args.model_name, args.lr))
        optimizer = optmizor.WTGD(model.parameters(), lr=args.lr, device=device)
    elif args.optimizer == 'Adam':
        writer = RecordWriter(
            'MNIST_{}_{}_{}_{}'.format(args.optimizer, args.batch_size, args.model_name, args.lr))
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        print('Optimizer not defined')
        exit()

    accs = []

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(args, model, device, train_loader, optimizer, epoch, LOSS_FUNC)
        test_loss, test_acc = test(args, model, device, test_loader, LOSS_FUNC)
        scheduler.step(epoch)
        accs.append(test_acc.avg)

        writer.update('train loss', train_loss.val)
        writer.update('test loss', test_loss.val)
        writer.update('train acc', train_acc.val)
        writer.update('test acc', test_acc.val)
        writer.update('test acc avg', [test_acc.avg])

        if args.save_model and test_acc.avg == max(accs):
            print("Model saved")
            # writer.update('test best acc', [test_acc.avg)
            torch.save(model.state_dict(), os.path.join(args.checkpoint_path,args.checkpoint_name))

    writer.write()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--optimizer', type=str, default='ARMAGD')
    parser.add_argument('--memory', nargs="+", type=float, default=[0.0, 0.9])
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='MOM',
                        help='momentum (default: 0.9)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--layers', type=list, default=[784, 10, 10],
                        help='The argument for defining the MLP')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--checkpoint-path', default='checkpoint/MNIST', type=str)
    parser.add_argument('--checkpoint-name', default='MLP_KO.pt', type=str)
    parser.add_argument('--model-name', default='MLP', type=str)
    parser.add_argument('--cuda', type=str, default='0')

    args = parser.parse_args()
    main(args)
