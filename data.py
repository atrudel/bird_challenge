from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import torchvision.transforms as transforms
from torchvision import datasets

# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 64 x 64 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# the training set

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.AugMix(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]),
    'val': transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
}

def view_class_imbalance(data_dir: str) -> None:
    dataset = datasets.ImageFolder(data_dir)
    labels: List[int] = dataset.targets
    class_counts: pd.Series = pd.Series(labels).value_counts()
    class_names: List[str] = dataset.classes
    plt.bar(x=class_names, height=class_counts)
    plt.title('Class count in the dataset')
    plt.xticks(fontsize=8,rotation=45, ha='right')
    plt.show()
