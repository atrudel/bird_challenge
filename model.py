import torch
import torch.nn as nn
from torchvision import models

nclasses = 20
IMSIZE = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get pretrained Imagenet model and freeze its parameters
pretrained_model = models.resnext50_32x4d(pretrained=True)
for param in pretrained_model.parameters():
    param.requires_grad = False

# Replace the classification head
nb_features = pretrained_model.fc.in_features
pretrained_model.fc = nn.Linear(nb_features, nclasses)


def load_model_and_unfreeze_parameters(model_path: str):
    state_dict = torch.load(model_path, map_location=device)
    pretrained_model.load_state_dict(state_dict)
    for param in pretrained_model.parameters():
        param.requires_grad = True
    return pretrained_model




if __name__ == '__main__':
    from pytorch_model_summary import summary
    print(summary(pretrained_model, torch.zeros(1, 3, IMSIZE, IMSIZE), show_input=False))