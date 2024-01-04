import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from aggregator import Aggregator
from frameworks import FedClient, FedServer
from models import SimpleCnn
from partitioner import DataPartitioner
from privacy import VanillaDp
from trainer import Trainer, ExpectationBasedTrainer
from torch import nn

device = 'cuda'

def plot_contrast(title, data, labels):
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    for i, datum in enumerate(data):
        plt.plot(datum, color=colors[i], label=labels[i], marker='.')
        plt.legend()
        plt.title(title)
    plt.savefig(f'./{title}.png')
    plt.close()

if __name__ == "__main__":
    n_client = 10
    n_round = 75
    n_epoch = 5
    privacy_budget = 1
    privacy_relax = 0.01
    lr = 0.01
    dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )

    testset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )

    clip_thr = {
        'conv1.weight': 5,
        'conv1.bias': 2,
        'conv2.weight': 5,
        'conv2.bias': 2,
        'fc1.weight': 5,
        'fc1.bias': 2,
        'fc2.weight': 5,
        'fc2.bias': 2,
        'fc3.weight': 5,
        'fc3.bias': 2,
    }

    testloader = DataLoader(testset, batch_size=64)

    partition = DataPartitioner(
        dataset,
        'dirichlet',
        n_client,
        {'show': True, 'alpha': 0.5, 'batch_size': 64, 'num_workers': 2}
    )

    protectors = [VanillaDp({
        'n_aggr': n_round,
        'n_dataset': len(partition[i].dataset),
        'epsilon': privacy_budget,
        'delta': privacy_relax,
        'clip_thr': clip_thr,
    }) for i in range(n_client)]

    titles = ['dp', 'dp_robust']
    acc_rec, loss_rec = [], []

    for lab in titles:
        FedClient.count = 0
        local_models = [SimpleCnn().to(device) for _ in range(n_client)]
        global_model = SimpleCnn().to(device)
        server = FedServer(
            global_model,
            Aggregator(global_model),
            partition.freq,
            {},
            testloader
        )

        if lab == 'baseline':
            clients = [FedClient(
                local_models[i],
                Trainer(
                    local_models[i],
                    partition[i],
                    nn.CrossEntropyLoss().to(device),
                    {'lr': lr}
                ),
                {'epoch': n_epoch, 'test': False, 'log': False},
                testloader,
            ) for i in range(n_client)]
        elif lab == 'robust':
            clients = [FedClient(
                local_models[i],
                ExpectationBasedTrainer(
                    local_models[i],
                    partition[i],
                    nn.CrossEntropyLoss().to(device),
                    {'lr': lr, 'std': protectors[i].std['conv1.weight']}
                ),
                {'epoch': n_epoch, 'test': False, 'log': False},
                testloader,
            ) for i in range(n_client)]
        elif lab == 'dp':
            clients = [FedClient(
                local_models[i],
                Trainer(
                    local_models[i],
                    partition[i],
                    nn.CrossEntropyLoss().to(device),
                    {'lr': lr}
                ),
                {'epoch': n_epoch, 'test': False, 'log': False},
                testloader,
                protectors[i]
            ) for i in range(n_client)]

        elif lab == 'dp-robust':
            clients = [FedClient(
                local_models[i],
                ExpectationBasedTrainer(
                    local_models[i],
                    partition[i],
                    nn.CrossEntropyLoss().to(device),
                    {'lr': lr, 'std': protectors[i].std['conv1.weight']}
                ),
                {'epoch': n_epoch, 'test': False, 'log': False},
                testloader,
                protectors[i]
            ) for i in range(n_client)]

        loss_li, acc_li = [], []
        for r in range(n_round):
            weights = {}
            for client in clients:
                client.train()
                weights[client.id] = client.get_weights()
                
            for item in weights[0].values():
                print(item.norm())
            server.aggregate(weights)
            loss, acc = server.log_test(r)
            loss_li.append(loss)
            acc_li.append(acc)
            glob_weight = server.model.state_dict()

            for client in clients:
                client.set_weights(glob_weight)

        acc_rec.append(acc_li)
        loss_rec.append(loss_li)


    acc_df = pd.DataFrame(acc_rec);
    loss_df = pd.DataFrame(loss_rec)
    acc_df.to_csv("./results/acc.csv")
    loss_df.to_csv("./results/loss.csv")

    plot_contrast("Accuracy", acc_rec, titles)
    plot_contrast("Loss", loss_rec, titles)