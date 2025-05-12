# -*- coding: utf-8 -*-
'''

Train CIFAR10 with PyTorch and Vision Transformers!
written by @kentaroy47, @arutema47

'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import csv
import time

from models import *
from utils import RandAugment, progress_bar
from dataset import getCIFAR10
from models.vit import ViT
from models.convmixer import ConvMixer

#####################################################################################################################
## System

datasetdir = os.getenv('TORCH_DATASETPATH', os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data'))
# savedir    = os.getenv('TORCH_TRAINPATH', os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data'))
savedir    = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

#####################################################################################################################
## Train Routine

# def train(epoch):
def train(net, epoch, workdir, netname, patch, best_acc):
    print('\nEpoch: %d' % epoch)

    ## Train Step
    net.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):

        inputs, targets = inputs.to(device), targets.to(device)

        # Train with amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += targets.size(0)
        train_correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*train_correct/train_total, train_correct, train_total))
    # return train_loss/(batch_idx+1)
    train_loss /= (batch_idx+1)

# def test(net, epoch, workdir, netname, patch):

    ## Validation
    net.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (val_loss/(batch_idx+1), 100.*val_correct/val_total, val_correct, val_total))
    
    acc = 100.*val_correct/val_total

    # Save checkpoint.
    if acc > best_acc:
        print('Saving checkpoint..')
        state = {'model': net.state_dict(),
            'acc': acc,
            'epoch': epoch}

        checkpointsdir = os.path.join(workdir, 'checkpoint')
        if not os.path.isdir(checkpointsdir):
            os.mkdir(checkpointsdir)
        
        torch.save(state, os.path.join(checkpointsdir, netname + f'-{patch}-ckpt.t7'))
        best_acc = acc
    
    ## Write Logs
    logdir = os.path.join(workdir, 'log')
    os.makedirs(logdir, exist_ok=True)

    # .txt Log
    if epoch == 0:
        with open(os.path.join(f'log_{args.net}_patch{args.patch}.txt'), 'w') as f:
            pass

    content = time.ctime() + f', Epoch: {epoch}, lr: {optimizer.param_groups[0]['lr']:.7f}, val loss: {val_loss:.5f}, acc: {(acc):.5f}'
    print(content)

    with open(os.path.join(logdir, 'log_' + args.net + '_patch' + str(patch) + '.txt'), 'a') as appender:
        appender.write(content + '\n')

    # .csv Log
    if epoch == 0:
        with open(os.path.join(logdir, f'log_{args.net}_patch{args.patch}.csv'), 'w') as f:
            pass

    with open(os.path.join(logdir, f'log_{args.net}_patch{args.patch}.csv'), 'a') as f:
        print(f'{train_loss}, {val_loss}, {acc}', file=f)

    return train_loss, val_loss, acc

#####################################################################################################################
## Networks

if __name__ == '__main__':

    workdir  = os.path.dirname(os.path.realpath(__file__))

    networks = [
        'res18',
        'alexnet',
        'vgg',
        'res34',
        'res50',
        'res101',
        'lenet',
        'vgg11',
        'convmixer',
        'mlpmixer',
        'vit_small',
        'vit_tiny',
        'simplevit',
        'vit',
        'vit_timm',
        'cait',
        'cait_small',
        'swin']

    # Args Parser
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4
    parser.add_argument('--opt', default='adam')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--noaug', action='store_false', help='disable use randomaug')
    parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
    parser.add_argument('--nowandb', action='store_true', help='disable wandb')
    parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
    parser.add_argument('--notrain', action='store_true', help='skips training')
    parser.add_argument('--net', choices=networks)
    parser.add_argument('--loadnet', type=str, help='load net')
    parser.add_argument('--dp', action='store_true', help='use data parallel')
    parser.add_argument('--batchsize', default=512)
    parser.add_argument('--size', default='32')
    parser.add_argument('--n_epochs', type=int, default='200')
    parser.add_argument('--patch', default='4', type=int, help='patch for ViT')
    parser.add_argument('--dimhead', default='512', type=int)
    parser.add_argument('--convkernel', default='8', type=int, help='parameter for convmixer')

    args = parser.parse_args()

    batchsize = int(args.batchsize)
    imsize = int(args.size)

    use_amp = not args.noamp
    aug = args.noaug

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    if args.net=='vit_timm':
        size = 384
    else:
        size = imsize

# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.Resize(size),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# transform_test = transforms.Compose([
#     transforms.Resize(size),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# # Add RandAugment with N, M(hyperparameter)
# if aug:  
#     N = 2
#     M = 14
#     transform_train.transforms.insert(0, RandAugment(N, M))

# Prepare dataset
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Model factory..
    print('==> Building model..')
    if args.net=='res18':
        net = ResNet18()
    elif args.net=='alexnet':
        net = AlexNet()
    elif args.net=='vgg':
        net = VGG('VGG19')
    elif args.net=='res34':
        net = ResNet34()
    elif args.net=='res50':
        net = ResNet50()
    elif args.net=='res101':
        net = ResNet101()
    elif args.net=='lenet':
        net = LeNet(dropout_value=0.5)
    elif args.net=='vgg11':
        net = VGG11CIFAR10(dropout_value=.1)
    elif args.net=='convmixer':
        # from paper, accuracy >96%. you can tune the depth and dim to scale accuracy and speed.
        net = ConvMixer(256, 16, kernel_size=args.convkernel, patch_size=1, n_classes=10)
    elif args.net=='mlpmixer':
        from models.mlpmixer import MLPMixer
        net = MLPMixer(
        image_size = 32,
        channels = 3,
        patch_size = args.patch,
        dim = 512,
        depth = 6,
        num_classes = 10
    )
    elif args.net=='vit_small':
        from models.vit_small import ViT
        net = ViT(
        image_size = size,
        patch_size = args.patch,
        num_classes = 10,
        dim = int(args.dimhead),
        depth = 6,
        heads = 8,
        mlp_dim = 512,
        dropout = 0.1,
        emb_dropout = 0.1
    )
    elif args.net=='vit_tiny':
        from models.vit_small import ViT
        net = ViT(
        image_size = size,
        patch_size = args.patch,
        num_classes = 10,
        dim = int(args.dimhead),
        depth = 4,
        heads = 6,
        mlp_dim = 256,
        dropout = 0.1,
        emb_dropout = 0.1
    )
    elif args.net=='simplevit':
        from models.simplevit import SimpleViT
        net = SimpleViT(
        image_size = size,
        patch_size = args.patch,
        num_classes = 10,
        dim = int(args.dimhead),
        depth = 6,
        heads = 8,
        mlp_dim = 512
    )
    elif args.net=='vit':
        # ViT for cifar10
        net = ViT(
        image_size = size,
        patch_size = args.patch,
        num_classes = 10,
        dim = int(args.dimhead),
        depth = 6,
        heads = 8,
        mlp_dim = 512,
        dropout = 0.1,
        emb_dropout = 0.1
    )
    elif args.net=='vit_timm':
        import timm
        net = timm.create_model('vit_base_patch16_384', pretrained=True)
        net.head = nn.Linear(net.head.in_features, 10)
    elif args.net=='cait':
        from models.cait import CaiT
        net = CaiT(
        image_size = size,
        patch_size = args.patch,
        num_classes = 10,
        dim = int(args.dimhead),
        depth = 6,   # depth of transformer for patch to patch attention only
        cls_depth=2, # depth of cross attention of CLS tokens to patch
        heads = 8,
        mlp_dim = 512,
        dropout = 0.1,
        emb_dropout = 0.1,
        layer_dropout = 0.05
    )
    elif args.net=='cait_small':
        from models.cait import CaiT
        net = CaiT(
        image_size = size,
        patch_size = args.patch,
        num_classes = 10,
        dim = int(args.dimhead),
        depth = 6,   # depth of transformer for patch to patch attention only
        cls_depth=2, # depth of cross attention of CLS tokens to patch
        heads = 6,
        mlp_dim = 256,
        dropout = 0.1,
        emb_dropout = 0.1,
        layer_dropout = 0.05
    )
    elif args.net=='swin':
        from models.swin import swin_t
        net = swin_t(window_size=args.patch,
                    num_classes=10,
                    downscaling_factors=(2,2,2,1))

    # For Multi-GPU
    if 'cuda' in device:
        print(device)
        if args.dp:
            print('using data parallel')
            net = torch.nn.DataParallel(net) # make parallel
            cudnn.benchmark = True


    #### Dataset
    trainloader, testloader, classes = getCIFAR10(datasetdir, size, args.batchsize, aug)


    #### Resume
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(os.path.join(workdir, 'checkpoint')), 'Error: no checkpoint directory found!'

        checkpoint = torch.load(os.path.join(workdir, 'checkpoint', f'{args.net}-{args.patch}-ckpt.t7'))
        net.load_state_dict(checkpoint['model'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    #### Training
    # Criterion
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    if args.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr)  
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)

    # Scaler
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    net.cuda()

    for epoch in range(start_epoch, args.n_epochs):
        start = time.time()
        train_loss, val_loss, acc = train(net, epoch, workdir, args.net, args.patch, best_acc)
        
        scheduler.step(epoch-1) # step cosine scheduling
        
        # print(f'{train_loss}, {val_loss}, {acc}')

    
