import torch
import torch.nn as nn
from torchvision import models
import timm

nclasses = 20
IMSIZE = 384

# Get pretrained Imagenet model and freeze its parameters
pretrained_model = timm.create_model('vit_base_patch16_384', pretrained=True)

# Replace the classification head
nb_features = list(pretrained_model.children())[-1].in_features
pretrained_model.head = nn.Identity()
pretrained_model.fc = nn.Linear(nb_features, nclasses)



if __name__ == '__main__':
    from pytorch_model_summary import summary
    print(summary(pretrained_model, torch.zeros(1, 3, IMSIZE, IMSIZE), show_input=False))