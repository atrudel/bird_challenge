import argparse
import os

import torch
import torch.optim as optim

from data import get_data_loader
from model import pretrained_model, load_model_and_unfreeze_parameters
from datetime import datetime

# Training settings
parser = argparse.ArgumentParser(description='RecVis A3 training script')
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=4, metavar='B',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='I',
                    help='how many batches to wait before logging training status')
parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
                    help='folder where experiment outputs are located.')
parser.add_argument('--name', type=str, default='exp', metavar='N', help='name of the experiment')
parser.add_argument('--scheduler-steps', type=int, default=7, metavar='K', help='number of steps before scheduler changes the lr')
parser.add_argument('--model', type=str, default='', help='Path to the pretrained model')


def setup_experiment(args) -> str:
    torch.manual_seed(args.seed)
    timestamp = datetime.now().strftime('%Y-%m-%d_%Hh%M')
    experiment_name = f"{args.name}_{timestamp}"
    experiment_path = f"{args.experiment}/{experiment_name}"
    # Create experiment folder
    if not os.path.isdir(args.experiment):
        print("Creating an experiment folder")
        os.makedirs(args.experiment)
    os.makedirs(experiment_path)
    print(f"Lauching experiment {experiment_name} for {args.epochs} epochs with LR={args.lr}, Momentum={args.momentum}, Schedule={args.scheduler_steps}steps")
    return experiment_path


def train(epoch: int, model, data_loader, optimizer, use_cuda: bool):
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
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
    with torch.set_grad_enabled(False):
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
    print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(data_loader.dataset), validation_accuracy))
    return validation_accuracy


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f'--> Saved model to {path}\n')


if __name__ == '__main__':
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    experiment_path = setup_experiment(args)
    train_loader = get_data_loader(args.data, 'train', args.batch_size)
    val_loader = get_data_loader(args.data, 'val', args.batch_size)

    # Neural network and optimizer
    # We define neural net in model.py so that it can be reused by the evaluate.py script
    if args.model == '':
        print("Using original checkpoint.")
        model = pretrained_model
    else:
        print(f"Finetuning a pre-trained checkpoint: {args.model}")
        model = load_model_and_unfreeze_parameters(args.model)
    if use_cuda:
        print('Using GPU\n')
        model.cuda()
    else:
        print('Using CPU\n')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_steps, gamma=0.1)

    best_val_accuracy = 0
    best_model_path = ''
    for epoch in range(1, args.epochs + 1):
        train(epoch, model, train_loader, optimizer, use_cuda)
        scheduler.step()
        val_accuracy = validation(model, val_loader)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_path = f"{experiment_path}/model_{epoch}(acc_{best_val_accuracy:.3g}).pth"
            save_model(model, path=best_model_path)
    print(f"The best model had a validation accuracy of {best_val_accuracy} and was saved in {best_model_path}\n"\
    f"You can run `python evaluate.py --model ' + path + '` to generate the Kaggle formatted csv file\n'")

