from typing import List

import pandas as pd
from matplotlib import pyplot as plt
from torchvision import datasets
import torch
import numpy as np

from data import get_data_loader, get_dataset
from model import pretrained_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def view_class_imbalance(data_dir: str) -> None:
    dataset = datasets.ImageFolder(data_dir)
    labels: List[int] = dataset.targets
    class_counts: pd.Series = pd.Series(labels).value_counts()
    class_names: List[str] = dataset.classes
    plt.bar(x=class_names, height=class_counts)
    plt.title('Class count in the dataset')
    plt.xticks(fontsize=8,rotation=45, ha='right')
    plt.show()

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)


def visualize_model(model, phase='val', num_images=6, errors_only=False):
    was_training = model.training
    model.eval()
    print("Loading data...")
    data_loader = get_data_loader('bird_dataset', phase, 103)
    class_names = get_dataset('bird_dataset', 'train').classes
    images_so_far = 0
    fig = plt.figure(figsize=(10, 2 * num_images))

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            print("Model inference...")
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct = preds == labels

            for j in range(inputs.size()[0]):
                if not errors_only or not correct[j]:
                    images_so_far += 1
                    ax = plt.subplot(num_images//2, 2, images_so_far)
                    ax.axis('off')
                    color = 'green'
                    title = f'[{j}] predicted: {class_names[preds[j]]}'
                    if not correct[j]:
                        color = 'red'
                        title += f"\ncorrect: {class_names[labels[j]]}"
                    ax.set_title(title, color=color, fontdict={'fontsize': 10})
                    imshow(inputs.cpu().data[j])

                if images_so_far > 0 and images_so_far % num_images == 0:
                    model.train(mode=was_training)
                    plt.tight_layout()
                    plt.show()
                    images_so_far = 0
                    text = input("Press enter to continue, q to quit")
                    if text == 'q':
                        return
        model.train(mode=was_training)


def view_confusion_matrix(model, phase='val'):
    was_training = model.training
    model.eval()
    print("Loading data...")
    data_loader = get_data_loader('bird_dataset', phase, 103)
    class_names = get_dataset('bird_dataset', 'train').classes

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            print("Model inference...")
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            confmat = confusion_matrix(labels, preds)
            display = ConfusionMatrixDisplay(confusion_matrix=confmat, display_labels=class_names)
            display.plot(xticks_rotation='vertical')

    model.train(mode=was_training)


if __name__ == '__main__':
    checkpoint = 'resultats/resnet18_2022-11-26_12h02/model_30(acc_86.4).pth'
    model = pretrained_model
    model.load_state_dict(torch.load(checkpoint, map_location=torch.device(device)))
    view_confusion_matrix(model, 'val')
    visualize_model(model, 'val')
