import copy
import math
import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import tenseal as ts
from aggregator import Aggregator, SplitAggregator
from frameworks import FedClient, FedServer
from models import SimpleCnn
from partitioner import DataPartitioner
from privacy import Ckks, CkksDp, PaillierDp, VanillaDp, Paillier
from trainer import Trainer, ExpectationBasedTrainer
from torch import nn
import time
from colorama import Fore, Style
import os
from utils import plot_contrast, bar_chart
from ipcl_python import PaillierKeypair
import utils
import training
from loguru import logger

device = 'cuda'
os.environ["CUDA_VISIBLE_DEVICES"]= "0,1"
logger.add("./results/progress.log", format="{time:YYYY-MM-DD HH:mm:ss} {level} {message}")

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
    clip_thr = {key: value * 0.3 for key, value in clip_thr.items()}
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
    
    dp_args_he = [{
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
    paillier_dp = [PaillierDp(pk, sk, None, dp_args_he[i]) for i in range(n_client)]
    
    # titles = ['he', 'dp', 'he-dp', 'baseline']
    titles = ['dp', 'he-dp', 'he']
    acc_rec, loss_rec, time_rec = [], [], []
    
    base_state = SimpleCnn().to(device).state_dict()
    global_model = SimpleCnn().to(device)
    for lab in titles:
        FedClient.count = 0
        global_model.load_state_dict(base_state)
        local_models = [SimpleCnn().to(device) for _ in range(n_client)]
        for model in local_models:
            model.load_state_dict(base_state)
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

            
        if lab == 'he-dp':
            rates = [0.1, 0.25, 0.5, 0.75]
            for rate in rates:
                global_model.load_state_dict(base_state)
                local_models = [SimpleCnn().to(device) for _ in range(n_client)]
                for model in local_models:
                    model.load_state_dict(global_model.state_dict())
                global_weight = {key: value.flatten() for key, value in global_model.state_dict().items()}
                        
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
                paillier_dp[i]
            ) for i in range(n_client)]
                
                start_time = time.time()
                init_he_idcs = {}
                for key, value in global_weight.items():
                    init_he_idcs[key] = np.random.choice(np.arange(0, len(value)), size=math.ceil(rate * len(value)), replace=False)
                loss_li, acc_li = training.he_dp_random_training(
                    clients=clients,
                    n_round=n_round,
                    he_frac=rate,
                    init_he_idcs=init_he_idcs,
                    freq=partition.freq
                )
                time_ = time.time() - start_time
                logger.success(f"Experiment: {lab}-{rate*100}%, Time: {time_} seconds")
                time_rec.append(time_)
                acc_rec.append(acc_li)
                loss_rec.append(loss_li)
            continue

        
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
        logger.success(f"Experiment: {lab}, Time: {time_} seconds")
        time_rec.append(time_)

    acc_df = pd.DataFrame(acc_rec)
    loss_df = pd.DataFrame(loss_rec)
    time_df = pd.DataFrame(time_rec)
    acc_df.to_csv("./results/acc.csv")
    loss_df.to_csv("./results/loss.csv")
    time_df.to_csv("./results/time.csv")
    
    image_titles = ['0%', '10%', '25%', '50%', '75%', '100%']
    plot_contrast("Accuracy", acc_rec, image_titles)
    plot_contrast("Loss", loss_rec, image_titles)
    bar_chart("Time", time_rec, image_titles)