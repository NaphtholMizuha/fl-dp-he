from torchvision import datasets, transforms

data_path = '/mnt/data/wuihou'
num_classes = {
    'mnist': 10,
    'fmnist': 10,
    'cifar10': 10,
    'cifar100': 100,
}

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
            
        elif name == 'cifar100':
            self.train = datasets.CIFAR100(
                root=data_path,
                train=True,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
            )
            
            self.test = datasets.CIFAR100(
                root=data_path,
                train=False,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
            )
            
        elif name == 'imagenet':
            self.train = datasets.ImageNet(
                root=data_path,
                split='train',
                download=True,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
            )
            self.test = datasets.ImageNet(
                root=data_path,
                split='val',
                download=True,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
            )