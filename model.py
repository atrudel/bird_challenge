import torch.nn as nn
from torchvision import models

nclasses = 20 

# Get pretrained Imagenet model and freeze its parameters
pretrained_model = models.resnet18(pretrained=True)
# for param in pretrained_model.parameters():
#     param.requires_grad = False

# Replace the classification head
nb_features = pretrained_model.fc.in_features
pretrained_model.fc = nn.Linear(nb_features, nclasses)