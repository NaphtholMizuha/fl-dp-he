import math
import numpy as np
import torch
from datasets import Datasets
from utils import get_top_k_idcs
from partitioner import DataPartitioner
from ipcl_python import PaillierKeypair
from privacy import Paillier, PaillierDp, VanillaDp
from models import model_select
from frameworks import FedClient, Aggregator, FedServer
from trainer import Trainer
from torch.utils.data import DataLoader
import torch.nn as nn
from loguru import logger
device = 'cuda'

def experiment(dataset_name: str, model_name: str):
    results = {}
    # set up the experiment
    dataset = Datasets(dataset_name)
    n_client = 10
    n_round = 30
    n_epoch = 3
    privacy_budget = 0.1
    privacy_relax = 1e-5
    lr = 0.01
    
    # partition the dataset
    partition = DataPartitioner(
        dataset.train,
        'dirichlet',
        n_client,
        {'show': True, 'alpha': 0.5, 'batch_size': 64, 'num_workers': 2}
    )
    testloader = DataLoader(dataset.test, batch_size=64)
    
    # set up the arguments for the differential privacy
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
    dp_args = [{
        'n_aggr': n_round,
        'n_dataset': len(partition[i].dataset),
        'epsilon': privacy_budget,
        'delta': privacy_relax,
        'clip_thr': clip_thr,
    } for i in range(n_client)]
    dp_list = [VanillaDp(dp_args[i]) for i in range(n_client)]
    
    # set up the arguments for the homomorphic encryption
    pk, sk = PaillierKeypair.generate_keypair(1024)
    paillier = Paillier(pk, sk)
    paillier_dp = [PaillierDp(pk, sk, None, dp_args[i]) for i in range(n_client)]
    
    # HE rates: 0 for all DP, 1 for all HE, and other values for mixed
    rates = [0, 0.1, 0.25, 0.5, 0.75, 1]
    
    for rate in rates:
        FedClient.count = 0
        logger.debug(f"Start the experiment at HE rates {rate*100}%")
        # create models and initialize loacl models with global model
        global_model = model_select(model_name).to(device)
        local_models = [model_select(model_name).to(device) for _ in range(n_client)]
        for model in local_models:
            model.load_state_dict(global_model.state_dict())
        w_glob = {key: value.flatten() for key, value in global_model.state_dict().items()}
        
        # create clients by HE rates
        if rate == 0:
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
        elif rate == 1:
            w_glob = paillier(w_glob)
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
        else:
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
            
        results[rate] = fed_train(clients, n_round, rate, w_glob, partition.freq)
    return results
        
            
def fed_train(clients: list[FedClient], n_round: int, he_rate: float, init_he_idcs: dict, freq: dict):
    """
    Train the clients with homomorphic encryption and differential privacy.
    """
    
    # initialize the loss and accuracy lists
    loss_li, acc_li = [], []
    
    # apply the HE indecies
    for client in clients:
        client.protector.he_idcs = init_he_idcs
    
    # global rounds
    for t in range(n_round):
        # local rounds
        for client in clients:
            # train local models
            client.train_update()
            
        # aggregate in DP-HE
        if he_rate != 0 and he_rate != 1:
            # he_updates and dp_updates are for two parts of updates, and top_idcs is for the top k indecies
            he_updates, dp_updates, top_idcs = [], [], []
            
            # client votes for the top k indecies by the raw updates
            for client in clients:
                tops = {}
                update = client.get_raw_update()
                for key, value in update.items():
                    tops[key] = get_top_k_idcs(value, math.ceil(he_rate * len(value)))
                top_idcs.append(tops)
            
            # aggregate the voting results
            idcs_glob = aggregate_he_idcs(top_idcs, he_rate)
        
            # apply voting results to the clients
            for client in clients:
                client.protector.he_idcs = idcs_glob
                he_update, dp_update = client.get_update()
                he_updates.append(he_update), dp_updates.append(dp_update)
            
            # aggregate the two parts of updates respectively
            he_glob = aggregate_updates(he_updates, freq)
            dp_glob = aggregate_updates(dp_updates, freq)
            
            # apply the global updates to the clients
            for client in clients:
                client.apply_updates((he_glob, dp_glob))
        # aggregate mannually
        else:
            updates = []
            for client in clients:
                updates.append(client.get_update())
            
            updates_glob = aggregate_updates(updates, freq)
            
            for client in clients:
                client.apply_updates(updates_glob)
            
        # test the new global model and record the loss and accuracy
        loss, acc = clients[0].log_test(t)
        loss_li.append(loss), acc_li.append(acc)
        
    return loss_li, acc_li

def aggregate_updates(updates, freq):
    updates_glob = {}
    for id, update in enumerate(updates):
        for key in update.keys():
            if key in updates_glob.keys():
                updates_glob[key] += update[key] * freq[id]
            else:
                updates_glob[key] = update[key] * freq[id]
                
    return updates_glob

def aggregate_he_idcs(he_idcs, he_frac):
    idcs_locals = {}
    idcs_glob = {}
    for he_idx in he_idcs:
        for key, value in he_idx.items():
            if key in idcs_locals.keys():
                idcs_locals[key] = torch.cat((idcs_locals[key], value))
            else:
                idcs_locals[key] = value
    for key, value in idcs_locals.items():
        counter = torch.bincount(value)
        idcs_glob[key] = get_top_k_idcs(counter, len(he_idcs[0][key]))
    return idcs_glob