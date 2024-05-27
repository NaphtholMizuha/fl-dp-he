import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, alexnet
from torchvision.models import mobilenet_v2

class TwoNN(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=[128, 64], output_dim=10):
        super().__init__()
        self.input_dim = input_dim
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )
        self.layer3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
    
class LeNet5Gray(nn.Module):
    def __init__(self, input_dim=256, hidden_dims=[120, 84], out_dim=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], out_dim)
        self.dropout = nn.Dropout(p=0.2)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
    
class LeNet5(nn.Module):
    def __init__(self, input_dim=400, hidden_dims=[120, 84], out_dim=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], out_dim)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=96,kernel_size=(3,3),stride=1,padding=1),nn.ReLU(),nn.MaxPool2d(kernel_size=(3,3),stride=2),
            nn.Conv2d(in_channels=96,out_channels=256,kernel_size=(5,5),stride=1,padding=2),nn.ReLU(),nn.MaxPool2d(kernel_size=(3,3),stride=2),
            nn.Conv2d(in_channels=256,out_channels=384,kernel_size=(3,3),padding=1),nn.ReLU(),
            nn.Conv2d(in_channels=384,out_channels=384,kernel_size=(3,3),padding=1),nn.ReLU(),
            nn.Conv2d(in_channels=384,out_channels=256,kernel_size=(3,3),padding=1),nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,3),stride=2)
                                    )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*3*3, 1024), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 10)
        )

    def forward(self,x):
        x = self.conv(x)
        x = self.fc(x)
        return x
    
# Define the Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


# Define the ResNet9
class ResNet9(nn.Module):
    def __init__(self, block, num_classes=10):
        super(ResNet9, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, stride=1)
        self.layer2 = self._make_layer(block, 128, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, block, out_channels, stride):
        layer = block(self.in_channels, out_channels, stride)
        self.in_channels = out_channels
        return layer

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
class MobileNetV2(nn.Module):
    def __init__(self, out_dim=10):
        super().__init__()
        self.mobilenet = mobilenet_v2(weights=None, num_classes=out_dim)

    def forward(self, x):
        x = self.mobilenet(x)
        return x
    
def model_select(name, num_classes=10) -> nn.Module:
    if name == '2nn':
        return TwoNN()
    elif name == 'cnn':
        return LeNet5Gray()
    elif name == 'lenet5':
        return LeNet5()
    elif name == 'alexnet':
        return AlexNet()
    elif name == 'resnet9':
        return ResNet9(ResidualBlock, num_classes=num_classes)
    elif name == 'mobilenetv2':
        return mobilenet_v2(weights=None, num_classes=num_classes)
    else:
        raise ValueError(f"Model {name} not found")