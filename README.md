# vision-transformers-cifar10
This is your go-to playground for training Vision Transformers (ViT) and its related models on CIFAR-10, a common benchmark dataset in computer vision.

The whole codebase is implemented in Pytorch, which makes it easier for you to tweak and experiment. Over the months, we've made several notable updates including adding different models like ConvMixer, CaiT, ViT-small, SwinTransformers, and MLP mixer. We've also adapted the default training settings for ViT to fit better with the CIFAR-10 dataset.

Using the repository is straightforward - all you need to do is run the `train_cifar10.py` script with different arguments, depending on the model and training parameters you'd like to use.

### Updates
* Added [ConvMixer]((https://openreview.net/forum?id=TVHS5Y4dNvM)) implementation. Really simple! (2021/10)

* Added wandb train log to reproduce results. (2022/3)

* Added CaiT and ViT-small. (2022/3)

* Added SwinTransformers. (2022/3)

* Added MLP mixer. (2022/6)

* Changed default training settings for ViT.

* Fixed some bugs and training settings (2024/2)

# Usage example
`python train_cifar10.py` # vit-patchsize-4

`python train_cifar10.py  --size 48` # vit-patchsize-4-imsize-48

`python train_cifar10.py --patch 2` # vit-patchsize-2

`python train_cifar10.py --net vit_small --n_epochs 400` # vit-small

`python train_cifar10.py --net vit_timm` # train with pretrained vit

`python train_cifar10.py --net convmixer --n_epochs 400` # train with convmixer

`python train_cifar10.py --net mlpmixer --n_epochs 500 --lr 1e-3`

`python train_cifar10.py --net cait --n_epochs 200` # train with cait

`python train_cifar10.py --net swin --n_epochs 400` # train with SwinTransformers

`python train_cifar10.py --net res18` # resnet18+randaug

# Results..

| MODELS  | MNIST | CIFAR10 | CIFAR100 | ImageNet | COCO |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Lenet  | 97.2  | 82.6  | x  | x  | x  |
| Alexnet | x  | 81.2  | 42.5  | 56.5  |x  |
| Res18 | x  | 94.5  | 71.6  | 69.7  | x  |
| Res34 | x  | 95.1  | 74.5  | 74.5  | x  |
| Res50 | x  | 95.4  | 74.6  | 80.3  | x  |
| Res101 | x  | x  | x  | 81.6  | x  |
| vgg11 | x  | 87.6  | x  | 69  | x  |
| vgg16 | x  | x  | x  | 71.5  | x  |
| vgg19 | x  | 93.5  | x  | 72.3  | x  |
| swin_t | x  | x  | 71.6  | 81  | x  |
| squeezenet | x  | x  | x  | 58.1  | x  |
| densenet | x  | x  | x  | 77.1  | x  |
| resnext50 | x  | x  | x  | 77.6  | x  |
| googlenet | x  | x  | x  | 69.7  | x  |
| mobilenetv2 | x  | x  | x  | 71.8  | x  |
| mobilenetv3 | x  | x  | x  | 74  | x  |
| yolov3 | x  | x  | x  | x  |   |
| ssdlite | x  | x  | x  | x  |   |


# Used in..
* Vision Transformer Pruning [arxiv](https://arxiv.org/abs/2104.08500) [github](https://github.com/Cydia2018/ViT-cifar10-pruning)
* Understanding why ViT trains badly on small datasets: an intuitive perspective [arxiv](https://arxiv.org/abs/2302.03751)
* Training deep neural networks with adaptive momentum inspired by the quadratic optimization [arxiv](https://arxiv.org/abs/2110.09057)
* [Moderate coreset: A universal method of data selection for real-world data-efficient deep learning ](https://openreview.net/forum?id=7D5EECbOaf9)
