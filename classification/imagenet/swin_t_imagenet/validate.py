import os
import argparse

import torch
from torch import optim
import torch.nn as nn

from imagenet import imageNET
from torchvision.models.swin_transformer import Swin_T_Weights
from torchvision.models import swin_t
from utils import test
from imports import load_ground_truth


if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    parser.add_argument('--valdir', type=str, required=True, help='Validation image directory')
    parser.add_argument('--vallist', type=str, required=True, help='Validation ground truth CSV file')
    parser.add_argument('--synset', type=str, required=True, help='Synset mapping file')
    parser.add_argument('--batchsize', type=int, default=64, help='Batch Size')

    args = parser.parse_args()

    # Load pretrained model
    print(f'-I({__file__}): Loading pretrained Swin Transformer model...')
    weights = Swin_T_Weights.DEFAULT
    model = swin_t(weights=weights)
    model.eval()
    print(f'-I({__file__}): Model loaded')

    # Load ground truth labels
    print(f'-I({__file__}): Loading ground truth labels...')
    image_ids, ground_truths = load_ground_truth(args.vallist)

    ## Dataset
    testloader = imageNET(args.valdir, args.vallist, args.synset, batchsize=args.batchsize)
    print(f'-I({__file__}): Evaluating Testing Accuracy...')
    test(testloader, model)