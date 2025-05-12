import os
import argparse

import torch
from torch import optim
import torch.nn as nn

from dataset import imageNET
import models
from utils import test
from imports import load_ground_truth


if __name__ == '__main__':

    networks = [
        'alexnet',
        'mobilenet',
        'resnet50',
        'resnet101',
        'swin_t'
    ]

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', choices=networks, required=True)
    parser.add_argument('--valdir', type=str, required=True, help='Validation image directory')
    parser.add_argument('--vallist', type=str, required=True, help='Validation ground truth CSV file')
    parser.add_argument('--synset', type=str, required=True, help='Synset mapping file')
    parser.add_argument('--batchsize', type=int, default=64, help='Batch Size')

    args = parser.parse_args()


    # Load pretrained model
    print(f'-I({__file__}): Loading model...')

    if args.model=='alexnet':
        model = models.AlexNet(weights=models.AlexNet_Weights)
    elif args.model=='mobilenet':
        model = models.MobileNetV3(weights=models.MobileNetV3_Weights)
    elif args.model=='resnet50':
        model = models.ResNet50(weights=models.ResNet50_Weights)
    elif args.model=='resnet101':
        model = models.ResNet101(weights=models.ResNet101_Weights)
    elif args.model=='swin_t':
        model = models.Swin_T(weights=models.Swin_T_Weights)
    else:
        raise ValueError('unrecognized model')


    model.eval()
    print(f'-I({__file__}): Model loaded')

    # Load ground truth labels
    print(f'-I({__file__}): Loading ground truth labels...')
    image_ids, ground_truths = load_ground_truth(args.vallist)

    ## Dataset
    testloader = imageNET(args.valdir, args.vallist, args.synset, batchsize=args.batchsize)
    print(f'-I({__file__}): Evaluating Testing Accuracy...')
    test(testloader, model)
