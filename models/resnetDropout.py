import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18CIFAR10(nn.Module):
    def __init__(self, dropout_value=0.5):
        super(ResNet18CIFAR10, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_value),
            nn.Linear(self.model.fc.in_features, 10)  # CIFAR-10 has 10 classes
        )
    
    def forward(self, x):
        x = self.model(x)
        return x
