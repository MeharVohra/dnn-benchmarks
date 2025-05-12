
import os
import argparse

import torch
from neural_compressor import PostTrainingQuantConfig, quantization
from torch import optim
import torch.nn as nn

from imagenet import imageNET
from torchvision.models.alexnet import AlexNet_Weights
from torchvision.models import alexnet
from utils import test
from imports import load_synset_mapping, load_ground_truth
from torch.utils.data import DataLoader

traindir   = os.environ['TORCH_TRAINDIR']
if __name__ == '__main__':

    savefile = os.path.join(traindir, 'fp32_alexnet_imagenet.pth')


    parser = argparse.ArgumentParser()

    parser.add_argument('--valdir', type=str, required=True, help='Validation image directory')
    parser.add_argument('--vallist', type=str, required=True, help='Validation ground truth CSV file')
    parser.add_argument('--synset', type=str, required=True, help='Synset mapping file')
    parser.add_argument('--batchsize', type=int, default=64, help='Batch Size')

    args = parser.parse_args()

    # # Load synset mapping
    # print(f'-I({__file__}): Loading synset mapping...')
    # synset_mapping = load_synset_mapping(args.synset)

    # Load pretrained model
    print(f'-I({__file__}): Loading pretrained AlexNet model...')
    weights = AlexNet_Weights.DEFAULT
    model = alexnet(weights=weights)
    model.eval()
    print(f'-I({__file__}): Model loaded')

    # Load ground truth labels
    print(f'-I({__file__}): Loading ground truth labels...')
    image_ids, ground_truths = load_ground_truth(args.vallist)

    ## Dataset
    testloader = imageNET(args.valdir, args.vallist, args.synset, batchsize=args.batchsize)
    print(f'-I({__file__}): Evaluating Testing Accuracy...')
    test(testloader, model)

    # 56.5

    # 56.5
    print('quantization begin!')
    model = alexnet().to(device='cpu')
    model.eval()

    def eval_func(model):
        # Use test set for evaluation
        acc, _ = test(testloader, model, complete=True, device='cpu', verbose=False)
        return acc

    class INCDataLoader:
        def __init__(self, dataloader, device = 'cpu'):
            self.dataloader = dataloader
            self.batch_size = dataloader.batch_size
            self.device = device
        def __iter__(self):
            for X, y in self.dataloader:
                yield X, y
        def __len__(self):
            return len(self.dataloader)


    conf = PostTrainingQuantConfig(approach='static')
    calib_dataloader = INCDataLoader(testloader)

    q_model = quantization.fit(
        model,
        conf=conf,
        calib_dataloader=calib_dataloader,
        eval_func=eval_func
    )


    print(f'-I({__file__}): Evaluating Quantized Model...')
    test(testloader, q_model)

    quant_model_path = os.path.join(traindir, 'int8Static_alexnet_imagenet.pth')
    torch.save(q_model.state_dict(), quant_model_path)


    print(f'-I({__file__}): Saved quantized model to {quant_model_path}')

