import math
import pandas as pd
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import tenseal as ts
from aggregator import Aggregator, SplitAggregator
from frameworks import FedClient, FedServer
from models import SimpleCnn
from partitioner import DataPartitioner
from privacy import Ckks, CkksDp, VanillaDp, Paillier
from trainer import Trainer, ExpectationBasedTrainer
from torch import nn
import time
from colorama import Fore, Style
import os
from utils import plot_contrast, bar_chart
from ipcl_python import PaillierKeypair
import utils

device = 'cuda'
os.environ["CUDA_VISIBLE_DEVICES"]= "0,1"


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
    
    ckks = Ckks(context)
    pk, sk = PaillierKeypair.generate_keypair(1024)
    paillier = Paillier(pk, sk)
    he_idcs = torch.load('./results/top_k_idcs_cpu.pt')
    ckks_dp = [CkksDp(context, he_idcs, dp_args[i]) for i in range(n_client)]
    
    # titles = ['he', 'dp', 'he-dp', 'baseline']
    titles = ['dp', 'baseline', 'he']
    acc_rec, loss_rec, time_rec = [], [], []

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
            global_weight = paillier(global_weight)
            server = FedServer(
                Aggregator(),
                partition.freq,
                'cpu',
                global_weight,
                {},
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
                paillier
            ) for i in range(n_client)]
        elif lab == 'he-dp':

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
                ckks_dp[i]
            ) for i in range(n_client)]

            he_weight = {}
            for key, value in global_weight.items():
                he_weight[key] = value.to('cpu')[he_idcs[key]]
            dp_weight = global_weight
            global_weight = (he_weight, dp_weight)

            server = FedServer(
            SplitAggregator(),
            partition.freq,
            'cuda',
            global_weight,
            {},
        )


        loss_li, acc_li = [], []
        start_time = time.time()
        for r in range(n_round):
            updates = {}
            for client in clients:
                client.train_update()
                update = client.get_update()
                updates[client.id] = update
                
            w_glob = server.aggregate_updates(updates)


            for client in clients:
                client.set_weights(w_glob)
                
            loss, acc = clients[0].log_test(r)
            loss_li.append(loss)
            acc_li.append(acc)

        acc_rec.append(acc_li)
        loss_rec.append(loss_li)
        time_ = time.time() - start_time
        print(f"{Fore.MAGENTA}Experiment: {lab}, Time: {time_} seconds{Style.RESET_ALL}")
        time_rec.append(time_)

    acc_df = pd.DataFrame(acc_rec)
    loss_df = pd.DataFrame(loss_rec)
    time_df = pd.DataFrame(time_rec)
    acc_df.to_csv("./results/acc.csv")
    loss_df.to_csv("./results/loss.csv")
    time_df.to_csv("./results/time.csv")

    plot_contrast("Accuracy", acc_rec, titles)
    plot_contrast("Loss", loss_rec, titles)
    bar_chart("Time", time_rec, titles)