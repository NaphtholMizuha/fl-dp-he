from torchvision import datasets, transforms

data_path = '/mnt/data/wuihou'

class Datasets:
    def __init__(self, name: str):
        if name == 'mnist':
            self.train = datasets.MNIST(
                root=data_path,
                train=True,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])
            )
            
            self.test = datasets.MNIST(
                root=data_path,
                train=False,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])
            )
            
        elif name == 'fmnist':
            self.train = datasets.FashionMNIST(
                root=data_path,
                train=True,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])
            )
            
            self.test = datasets.FashionMNIST(
                root=data_path,
                train=False,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])
            )
            
        elif name == 'cifar10':
            self.train = datasets.CIFAR10(
                root=data_path,
                train=True,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
            )
            self.test = datasets.CIFAR10(
                root=data_path,
                train=False,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
            )