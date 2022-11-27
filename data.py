import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets

from model import IMSIZE

# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 64 x 64 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# the training set

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation((-30, 30)),
        transforms.RandomPerspective(distortion_scale=0.3),
        transforms.RandomResizedCrop(224, scale=(0.4, 0.9), ratio=(1,1)),
        # transforms.Resize(IMSIZE),
        # transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(0.1),
        transforms.GaussianBlur(9, (0.1, 2)),
        transforms.RandomEqualize(p=0.2),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),

        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]),
    'val': transforms.Compose([
        transforms.Resize(IMSIZE),
        transforms.CenterCrop(224),
        transforms.GaussianBlur(5, (0.1, 2)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
}

# class RandomOcclusions(torch.nn.Module):
#     def __init__(self, n, min_size, max_size):
#         super().__init__()
#

def get_dataset(folder: str, phase:str):
    assert phase in ['train', 'val'], "Phase must be either 'train' or 'val'"
    return datasets.ImageFolder(
        root=f"{folder}/{phase}_images",
        transform=data_transforms[phase]
    )
def get_data_loader(directory: str, phase: str, batch_size: int, num_workers=2) -> DataLoader:
    assert phase in ['train', 'val'], "Phase must be either 'train' or 'val'"
    dataset = get_dataset(directory, phase)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(phase == 'train'),
        num_workers=num_workers
    )
    return data_loader

def visualize_dataset(phase: str):
    from visualization import imshow
    num_images = 6
    data_loader = get_data_loader('bird_dataset', phase, num_images, num_workers=1)
    images_so_far = 0
    fig = plt.figure(figsize=(10, 2 * num_images))

    for i, (inputs, labels) in enumerate(data_loader):
        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images // 2, 2, images_so_far)
            ax.axis('off')
            imshow(inputs.cpu().data[j])

            if images_so_far > 0 and images_so_far % num_images == 0:
                plt.tight_layout()
                plt.show()
                images_so_far = 0
                text = input("Press enter to continue, q to quit")
                if text == 'q':
                    return
if __name__ == '__main__':
    visualize_dataset('train')
