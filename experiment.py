import math
import os
import random
import numpy as np
import torch
from datasets import Datasets
from utils import get_bottom_k_idcs, get_top_k_idcs, get_bottom_k_norm
from partitioner import DataPartitioner
from ipcl_python import PaillierKeypair
from privacy import Paillier, PaillierDp, VanillaDp
from models import model_select
from frameworks import FedClient, Aggregator, FedServer
from trainer import Trainer
from torch.utils.data import DataLoader
import torch.nn as nn
from loguru import logger
import pickle
device = 'cuda'

flag = True

class Experiment:
    
    clip_default = {}
    clip_default['lenet5'] = {
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
    clip_default['2nn'] = {
        'layer1.0.weight': 1,
        'layer1.0.bias': 1,
        'layer2.0.weight': 1,
        'layer2.0.bias': 1,
        'layer3.weight': 1,
        'layer3.bias': 1
    }
    def __init__(self,
                    dataset_name: str, model_name: str,
                    n_client: int=10,
                    n_round: int=30,
                    n_epoch: int=3,
                    epsilon: float=0.1,
                    delta: float=1e-5,
                    clip: dict=None,
                    lr: float=0.01) -> None:
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.n_client = n_client
        self.n_round = n_round
        self.n_epoch = n_epoch
        self.epsilon = epsilon
        self.delta = delta
        self.lr = lr
        
        if clip is None and self.clip_default.get(model_name) is not None:
            self.clip = self.clip_default[model_name]
        else:
            self.clip = clip
        pass
    
    def run_by_rates(self, rates: list):
            results = {}
            loss_rec, acc_rec = [], []
            # set up the experiment
            dataset = Datasets(self.dataset_name)
            n_client = self.n_client
            n_round = self.n_round
            n_epoch = self.n_epoch
            privacy_budget = self.epsilon
            privacy_relax = self.delta
            lr = self.lr
            # HE rates: 0 for all DP, 1 for all HE, and other values for mixed

            # partition the dataset

            partition = DataPartitioner(
                dataset.train,
                'dirichlet',
                n_client,
                {'show': True, 'alpha': 0.5, 'batch_size': 64, 'num_workers': 2}
            )
            testloader = DataLoader(dataset.test, batch_size=64)
            

            # set up the arguments for the homomorphic encryption
            pk, sk = PaillierKeypair.generate_keypair(1024)
            paillier = Paillier(pk, sk)
            
            init_model = model_select(self.model_name).to(device)
            
            if self.clip is None:
                self.clip = {key: 0.01 for key in init_model.state_dict().keys()}
                
            w_init = {key: value.flatten() for key, value in init_model.state_dict().items()}
            for rate in rates:
                FedClient.count = 0
                logger.debug(f"Start the experiment at HE rates {rate*100}%")
                # create models and initialize loacl models with global model
                local_models = [model_select(self.model_name).to(device) for _ in range(n_client)]
                for model in local_models:
                    model.load_state_dict(init_model.state_dict())
                    
                if rate != 1:
                    dp_args = [{
                        'n_aggr': n_round,
                        'n_dataset': len(partition[i].dataset),
                        'epsilon': privacy_budget,
                        'delta': privacy_relax,
                        'clip_thr': self.clip,
                    } for i in range(n_client)]
                    dp_list = [VanillaDp(dp_args[i]) for i in range(n_client)]
                    paillier_dp = [PaillierDp(pk, sk, None, dp_args[i]) for i in range(n_client)]
                
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
                    w_init = paillier(w_init)
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
                    
                loss, acc = fed_train(clients, n_round, rate, w_init, partition.freq)
                loss_rec.append(loss), acc_rec.append(acc)
            return loss_rec, acc_rec
        
    def run_by_basis(self, basis: list, rate: float):
        results = {}
        loss_rec, acc_rec = [], []
        # set up the experiment
        dataset = Datasets(self.dataset_name)

        partition = DataPartitioner(
            dataset.train,
            'dirichlet',
            self.n_client,
            {'show': True, 'alpha': 0.5, 'batch_size': 64, 'num_workers': 2}
        )

        testloader = DataLoader(dataset.test, batch_size=64)
            
        
        # set up the arguments for the homomorphic encryption
        pk, sk = PaillierKeypair.generate_keypair(1024)    
        init_model = model_select(self.model_name).to(device)
        if self.clip is None:
            self.clip = {key: 1 for key in init_model.state_dict().keys()}
        w_init = {key: value.flatten() for key, value in init_model.state_dict().items()}
        for base in basis:
            FedClient.count = 0
            logger.debug(f"Start the experiment at base '{base}'")
            # create models and initialize loacl models with global model
            local_models = [model_select(self.model_name).to(device) for _ in range(self.n_client)]
            for model in local_models:
                model.load_state_dict(init_model.state_dict())
                
            dp_args = [{
                'n_aggr': self.n_round,
                'n_dataset': len(partition[i].dataset),
                'epsilon': self.epsilon,
                'delta': self.delta,
                'clip_thr': self.clip,
            } for i in range(self.n_client)]
            dp_list = [VanillaDp(dp_args[i]) for i in range(self.n_client)]
            paillier_dp = [PaillierDp(pk, sk, None, dp_args[i]) for i in range(self.n_client)]
            
            
            clients = [FedClient(
                local_models[i],
                Trainer(
                    local_models[i],
                    partition[i],
                    nn.CrossEntropyLoss().to(device),
                    {'lr': self.lr}
                ),
                {'epoch': self.n_epoch, 'test': False, 'log': False},
                testloader,
                paillier_dp[i]
            ) for i in range(self.n_client)]
                
            loss, acc = fed_train(clients, self.n_round, rate, w_init, partition.freq, vote=base)
            loss_rec.append(loss), acc_rec.append(acc)
        return loss_rec, acc_rec
        

def config_thresholds(model_name: str, n_client: int, n_round: int, n_epoch: int, partition: DataPartitioner, rates: list):
    model = model_select(model_name).to(device)
    keys = model.state_dict().keys()
    
    thresholds = {rate: [
        {key: 0.0 for key in keys} for _ in range(n_client)
    ] for rate in rates}
    
    clients = [FedClient(
        model,
        Trainer(
            model,
            partition[i],
            nn.CrossEntropyLoss().to(device),
            {'lr': 0.01}
        ),
        {'epoch': n_epoch, 'test': False, 'log': False},
    ) for i in range(n_client)]
    
    for t in range(n_round):
        for client in clients:
            client.train_update()
            update = client.get_raw_update()
            for rate in rates:
                for key in keys:
                    data = update[key].to('cpu')
                    thresholds[rate][client.id][key] += get_bottom_k_norm(data, math.floor((1 - rate) * len(data))) / n_round
        logger.debug(f"Tuning C: Round {t} is done.")
    return thresholds

def fed_train(clients: list[FedClient], n_round: int, he_rate: float, init_he_idcs: dict, freq: dict, vote: str="best"):
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
            tops = {}
            
            if vote != "random":
                # client votes for the top k indecies by the raw updates
                for client in clients:
                    
                    update = client.get_raw_update()
                    for key, value in update.items():
                        if vote == "best": 
                            tops[key] = get_top_k_idcs(value.abs(), math.ceil(he_rate * len(value)))
                        elif vote == "worst":
                            tops[key] = get_bottom_k_idcs(value.abs(), math.ceil(he_rate * len(value)))
                    top_idcs.append(tops)
                
                # aggregate the voting results
                idcs_glob = aggregate_votes(top_idcs)
            else:
                idcs_glob = {}
                update = clients[0].get_raw_update()
                for key, value in update.items():
                    idcs_glob[key] = torch.LongTensor(random.choices(range(len(value)), k=math.ceil(he_rate * len(value))))
        
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

def aggregate_votes(he_idcs):
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