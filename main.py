import argparse
import os

import torch
import torch.optim as optim
from torchvision import datasets
from model import Net

# Training settings
parser = argparse.ArgumentParser(description='RecVis A3 training script')
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='B',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
                    help='folder where experiment outputs are located.')

def setup_experiment(args):
    torch.manual_seed(args.seed)
    # Create experiment folder
    if not os.path.isdir(args.experiment):
        os.makedirs(args.experiment)

def get_data_loaders(args):
    from data import data_transforms

    # Data initialization and loading
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + '/train_images',
                             transform=data_transforms['train']),
        batch_size=args.batch_size, shuffle=True, num_workers=1)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + '/val_images',
                             transform=data_transforms['val']),
        batch_size=args.batch_size, shuffle=False, num_workers=1)

    return train_loader, val_loader



def train(epoch: int, model, data_loader, optimizer, use_cuda: bool):
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.data.item()))

def validation(model, data_loader) -> float:
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in data_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        validation_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(data_loader.dataset)
    validation_accuracy = 100. * correct / len(data_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        validation_loss, correct, len(data_loader.dataset), validation_accuracy))
    return validation_accuracy


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print('Saved model to ' + path + '. You can run `python evaluate.py --model ' + path + '` to generate the Kaggle formatted csv file\n')


if __name__ == '__main__':
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    setup_experiment(args)
    train_loader, val_loader = get_data_loaders(args)

    # Neural network and optimizer
    # We define neural net in model.py so that it can be reused by the evaluate.py script
    model = Net()
    if use_cuda:
        print('Using GPU')
        model.cuda()
    else:
        print('Using CPU')

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    best_val_accuracy = 0
    best_model_path = ''
    for epoch in range(1, args.epochs + 1):
        train(epoch, model, train_loader, optimizer, use_cuda)
        val_accuracy = validation(model, val_loader)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_path = f"{args.experiment}/model_{epoch}(acc_{best_val_accuracy:.3f}).pth"
            save_model(model, path=best_model_path)
    print(f"The best model had a validation accuracy of {best_val_accuracy} and was saved in {best_model_path}")

