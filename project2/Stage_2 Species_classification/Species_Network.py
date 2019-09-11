import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(3, 6, 3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(6 * 123 * 123, 150)
        self.relu3 = nn.ReLU(inplace=True)

        self.drop = nn.Dropout2d()

        self.fc2 = nn.Linear(150, 3)
        self.softmax1 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.relu2(x)

        # print(x.shape)
        x = x.view(-1, 6 * 123 * 123)
        x = self.fc1(x)
        x = self.relu3(x)

        x = F.dropout(x, training=self.training)

        x_species = self.fc2(x)
        x_species = self.softmax1(x_species)

        return x_species


class Net_morelayer(nn.Module):
    def __init__(self):
        super(Net_morelayer, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(3, 6, 3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(6, 12, 3)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.relu3 = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(6 * 60 * 60, 150)
        self.relu3 = nn.ReLU(inplace=True)

        self.drop = nn.Dropout2d()

        self.fc2 = nn.Linear(150, 3)
        self.softmax1 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.relu3(x)

        # print(x.shape)
        x = x.view(-1, 6 * 60 * 60)
        x = self.fc1(x)
        x = self.relu3(x)

        x = F.dropout(x, training=self.training)

        x_species = self.fc2(x)
        x_species = self.softmax1(x_species)

        return x_species


class Net_withnorm(nn.Module):
    def __init__(self):
        super(Net_withnorm, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 3, bias=False)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(3, 6, 3, bias=False)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(6 * 123 * 123, 150)
        self.relu3 = nn.ReLU(inplace=True)

        self.drop = nn.Dropout2d()

        self.fc2 = nn.Linear(150, 3)
        self.softmax1 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = nn.BatchNorm2d(3,track_running_stats=self.train)(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = nn.BatchNorm2d(6,track_running_stats=self.train)(x)
        x = self.relu2(x)

        # print(x.shape)
        x = x.view(-1, 6 * 123 * 123)
        x = self.fc1(x)
        x = self.relu3(x)

        x = F.dropout(x, training=self.training)

        x_species = self.fc2(x)

        return x_species
