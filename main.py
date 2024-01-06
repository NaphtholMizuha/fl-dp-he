import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import tenseal as ts
from aggregator import Aggregator
from frameworks import FedClient, FedServer
from models import SimpleCnn
from partitioner import DataPartitioner
from privacy import CkksHe, CkksDp, VanillaDp
from trainer import Trainer, ExpectationBasedTrainer
from torch import nn
import time
from colorama import Fore, Style

device = 'cuda'

# TODO: 将整个学习流程包装成函数，有上传模型和上传更新两个版本

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
    n_round = 30
    n_epoch = 3
    privacy_budget = 0.1
    privacy_relax = 1e-5
    lr = 0.01
    clip_thr = {
        'conv1.weight': 1,
        'conv1.bias': 0.3,
        'conv2.weight': 1,
        'conv2.bias': 0.3,
        'fc1.weight': 1,
        'fc1.bias': 0.3,
        'fc2.weight': 1,
        'fc2.bias': 0.3,
        'fc3.weight': 1,
        'fc3.bias': 1.1,
    }
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
    
    # Setup TenSEAL context
    context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=8192,
                coeff_mod_bit_sizes=[60, 40, 40, 60]
            )
    context.generate_galois_keys()
    context.global_scale = 2**40

    testloader = DataLoader(testset, batch_size=64)

    partition = DataPartitioner(
        dataset,
        'dirichlet',
        n_client,
        {'show': True, 'alpha': 0.5, 'batch_size': 64, 'num_workers': 2}
    )

    dp_args = [{
        'n_aggr': n_round,
        'n_dataset': len(partition[i].dataset),
        'epsilon': privacy_budget,
        'delta': privacy_relax,
        'clip_thr': clip_thr,
    } for i in range(n_client)]

    dp_list = [VanillaDp(dp_args[i]) for i in range(n_client)]
    
    ckks = CkksHe(context)
    ckks_dp = [CkksDp(context, [0], dp_args[i]) for i in range(n_client)]
    
    titles = ['baseline']
    acc_rec, loss_rec = [], []

    for lab in titles:
        FedClient.count = 0
        local_models = [SimpleCnn().to(device) for _ in range(n_client)]
        global_model = SimpleCnn().to(device)
        global_weight = {key: value.flatten() for key, value in global_model.state_dict().items()}
        server = FedServer(
            Aggregator(),
            partition.freq,
            'cuda',
            global_weight,
            {},
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
                dp_list[i]
            ) for i in range(n_client)]
        elif lab == 'he':
            server = FedServer(
                global_model,
                Aggregator(global_model),
                partition.freq,
                'cpu',
                {},
                testloader
            )
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
                ckks
            ) for i in range(n_client)]
        elif lab == 'he-dp':
            server = FedServer(
                global_model,
                Aggregator(global_model),
                partition.freq,
                'cpu',
                {},
                testloader
            )
            clients = [FedClient(
                local_models[i],
                Trainer(
                    local_models[i],
                    partition[i],
                    nn.CrossEntropyLoss().to(device),
                    {'lr': lr}
                ),
                {'epoch': n_epoch, 'test': False, 'log': True},
                testloader,
                ckks_dp[i]
            ) for i in range(n_client)]


        loss_li, acc_li = [], []
        start_time = time.time()
        for r in range(n_round):
            updates = {}
            for client in clients:
                client.train_update()
                updates[client.id] = client.get_update()
            w_glob = server.aggregate_updates(updates)


            for client in clients:
                client.set_weights(w_glob)
                
            loss, acc = clients[0].log_test(r)
            loss_li.append(loss)
            acc_li.append(acc)

        acc_rec.append(acc_li)
        loss_rec.append(loss_li)
        print(f"{Fore.MAGENTA}Experiment: {lab}, Time: {time.time() - start_time} seconds{Style.RESET_ALL}")

    acc_df = pd.DataFrame(acc_rec)
    loss_df = pd.DataFrame(loss_rec)
    acc_df.to_csv("./results/acc.csv")
    loss_df.to_csv("./results/loss.csv")

    plot_contrast("Accuracy", acc_rec, titles)
    plot_contrast("Loss", loss_rec, titles)