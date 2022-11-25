import torch
import torch.nn as nn
from torchvision import models

from data import RESIZE

nclasses = 20 

# Get pretrained Imagenet model and freeze its parameters
pretrained_model = models.inception_v3(pretrained=True)
for param in pretrained_model.parameters():
    param.requires_grad = False

# Replace the classification head
nb_features = pretrained_model.fc.in_features
pretrained_model.fc = nn.Linear(nb_features, nclasses)


if __name__ == '__main__':
    from pytorch_model_summary import summary
    print(summary(pretrained_model, torch.zeros(1, 3, RESIZE, RESIZE), show_input=False))