import torch
from torch import nn
import torchvision
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class CNN(nn.Module):
    def __init__(self,
                in_channels: int,
                class_num: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                            out_channels=16,
                            kernel_size=3,
                            padding=1
                            )
        self.conv2 = nn.Conv2d(in_channels=16,
                            out_channels=32,
                            kernel_size=3,
                            padding=1
                            )
        self.conv3 = nn.Conv2d(in_channels=32,
                            out_channels=64,
                            kernel_size=3,
                            padding=1
                            )

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc1 = nn.Linear(64 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, class_num)

        self.relu = nn.ReLU()

    def forward(self, x):
        # Convolution, relu, and pooling
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def display_img(dataset, index):
    img, label = dataset[index], dataset[index][1]
    plt.imshow(img[0].squeeze(), cmap='gray')
    plt.title(f'Label: {label}')
    plt.show


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn):
    epochs = 5
    loss_list = []
    accuracy_list = []

    loss_list = []

    model.train()
    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0
        for batch, (X,y) in enumerate(data_loader):
            y_pred = model(X)

            loss = loss_fn(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss
            batch_acc = accuracy_fn(y_pred.argmax(dim=1), y)
            train_acc += accuracy_fn(y_true=y,
                                    y_pred=y_pred.argmax(dim=1))

            loss_list.append(loss.item())
            accuracy_list.append(batch_acc)

            if batch % 50 == 0:
                print(f'Epoch: {epoch}, Batch: {batch}, Loss: {loss.item():.4f}, Accuracy: {batch_acc:.4f}')

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(loss_list)
    plt.title("Training Loss")
    plt.xlabel('Batch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(range(len(accuracy_list)), accuracy_list)
    plt.title("Training Accuracy")
    plt.xlabel('Batch')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.show()


def test_step(model: torch.nn.Module,
                data_loader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                accuracy_fn):
    epochs = 5
    test_loss, test_acc = 0, 0

    
    model.eval()
    with torch.inference_mode():
        for epoch in range(epochs):
            test_loss = 0
            test_acc = 0
            for (X,y) in data_loader:

                y_pred = model(X)

                loss = loss_fn(y_pred, y)

                test_loss += loss
                test_acc += accuracy_fn(y_true=y,
                                        y_pred=y_pred.argmax(dim=1))

    test_loss /= len(data_loader)
    test_acc /= len(data_loader)
    print(f"\nTest Loss: {test_loss:.5f} | Test acc: {test_acc:.2f}")



if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,))])

    train_data = datasets.MNIST(root='./data',
                                train=True,
                                download=True,
                                transform=transform)
    test_data = datasets.MNIST(root='./data',
                            train=False,
                            download=True,
                            transform=transform)

    # we can't give the computer all the data at once so we give it 64 at a time.
    # we then shuffle it to better help it learn.
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

    model = CNN(1, 10)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    train_step(model, train_loader, loss_fn, optimizer, accuracy_fn)
    test_step(model, test_loader, loss_fn, accuracy_fn)