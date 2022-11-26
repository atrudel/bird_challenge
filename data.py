import torch
import torchvision.transforms as transforms
from torchvision import datasets
from model import IMSIZE
from torch.utils.data import DataLoader

# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 64 x 64 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# the training set

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]),
    'val': transforms.Compose([
        transforms.Resize(IMSIZE),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
}

def get_dataset(folder: str, phase:str):
    assert phase in ['train', 'val'], "Phase must be either 'train' or 'val'"
    return datasets.ImageFolder(
        root=f"{folder}/{phase}_images",
        transform=data_transforms[phase]
    )
def get_data_loader(directory: str, phase: str, batch_size: int) -> DataLoader:
    assert phase in ['train', 'val'], "Phase must be either 'train' or 'val'"
    dataset = get_dataset(directory, phase)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(phase == 'train'),
        num_workers=4
    )
    return data_loader


