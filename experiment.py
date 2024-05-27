import math
import os
import random
import time
import numpy as np
import torch
from datasets import Datasets, num_classes
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
    def __init__(self, args):
        self.dataset_name = args.data
        self.model_name = args.model
        self.n_client = args.clients
        self.n_round = args.rounds
        self.n_epoch = args.epochs
        self.epsilon = args.epsilon
        self.delta = args.delta
        self.lr = args.lr
        self.log_client = args.log_client
        self.test_client = args.test_client
        self.args = args
        if self.clip_default.get(self.model_name) is not None:
            self.clip = self.clip_default[self.model_name]
            
    def run_by_clips(self, clips: list, repeat: int):
        loss_rec, acc_rec, time_rec = [], [], []
        rate = 0.1
        
        
        dataset = Datasets(self.dataset_name)
        
        
        if os.path.exists(f"./temp/{self.dataset_name}-{repeat}.npy"):
            partition = DataPartitioner(
                dataset.train,
                'load',
                self.n_client,
                {'show': False, 'alpha': self.args.alpha, 'batch_size': 64, 'num_workers': 2, 'load': f"./temp/{self.dataset_name}-{repeat}.npy"}
            )
        else:
            partition = DataPartitioner(
                dataset.train,
                self.args.partition,
                self.n_client,
                {'show': False, 'alpha': self.args.alpha, 'batch_size': 64, 'num_workers': 2, 'save': f"./temp/{self.dataset_name}-{repeat}.npy"}
            )
        testloader = DataLoader(dataset.test, batch_size=64)
        pk, sk = PaillierKeypair.generate_keypair(1024)
        paillier = Paillier(pk, sk)
        init_model = model_select(self.model_name, num_classes=num_classes[self.dataset_name]).to(device)
        if not hasattr(self, 'clip'):
            self.clip = {key: 1 for key in init_model.state_dict().keys()}
            
        w_init = {key: value.flatten() for key, value in init_model.state_dict().items()}
        
        for clip in clips:
            FedClient.count = 0
            logger.debug(f"Start the experiment at clip threshold scale '{clip}'")
            # create models and initialize loacl models with global model
            local_models = [model_select(self.model_name, num_classes=num_classes[self.dataset_name]).to(device) for _ in range(self.n_client)]
            for model in local_models:
                model.load_state_dict(init_model.state_dict())
                
            dp_args = [{
                'n_aggr': self.n_round,
                'n_dataset': len(partition[i].dataset),
                'epsilon': self.epsilon,
                'delta': self.delta,
                'clip_thr': {key: value * clip for key, value in self.clip.items()},
            } for i in range(self.n_client)]
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
                
            time_start = time.time()
            init_idcs = {}
            for key, value in w_init.items():
                init_idcs[key] = torch.randperm(len(value))[:math.ceil(rate * len(value))]
            # print(init_idcs)
            loss, acc = fed_train(clients, self.n_round, rate, init_idcs, partition.freq)
            time_rec.append(time.time() - time_start)
            loss_rec.append(loss), acc_rec.append(acc)
        return loss_rec, acc_rec, time_rec
    
    def run_by_rates(self, rates: list, repeat: int):
        results = {}
        loss_rec, acc_rec = [], []
        time_rec = []
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
        
        if os.path.exists(f"./temp/{self.dataset_name}-{repeat}.npy"):
            partition = DataPartitioner(
                dataset.train,
                'load',
                n_client,
                {'show': False, 'alpha': self.args.alpha, 'batch_size': 64, 'num_workers': 2, 'load': f"./temp/{self.dataset_name}-{repeat}.npy"}
            )
        else:
            partition = DataPartitioner(
                dataset.train,
                self.args.partition,
                n_client,
                {'show': False, 'alpha': self.args.alpha, 'batch_size': 64, 'num_workers': 2, 'save': f"./temp/{self.dataset_name}-{repeat}.npy"}
            )
        testloader = DataLoader(dataset.test, batch_size=64)
        

        # set up the arguments for the homomorphic encryption
        pk, sk = PaillierKeypair.generate_keypair(1024)
        paillier = Paillier(pk, sk)
        
        init_model = model_select(self.model_name, num_classes=num_classes[self.dataset_name]).to(device)
        
        if not hasattr(self, 'clip'):
            self.clip = {key: 1 for key in init_model.state_dict().keys()}
            
        w_init = {key: value.flatten() for key, value in init_model.state_dict().items()}
        for rate in rates:
            FedClient.count = 0
    

            logger.debug(f"Start the experiment at HE rates {rate*100}%")
            # create models and initialize local models with global model
            local_models = [
                model_select(self.model_name, num_classes=num_classes[self.dataset_name]).to(device)
                for _ in range(n_client)]
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
                    {'epoch': n_epoch, 'test': self.test_client, 'log': self.log_client},
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
                    {'epoch': n_epoch, 'test': self.test_client, 'log': self.log_client},
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
                    {'epoch': n_epoch, 'test': self.test_client, 'log': self.log_client},
                    testloader,
                    paillier_dp[i]
                ) for i in range(n_client)]
                
            time_start = time.time()
            loss, acc = fed_train(clients, n_round, rate, w_init, partition.freq)
            time_rec.append(time.time() - time_start)
            loss_rec.append(loss), acc_rec.append(acc)
        return loss_rec, acc_rec, time_rec
        
    def run_by_basis(self, basis: list, rate: float, repeat: int):
        results = {}
        loss_rec, acc_rec, time_rec = [], [], []
        # set up the experiment
        dataset = Datasets(self.dataset_name)

        if os.path.exists(f"./temp/{self.dataset_name}-{repeat}.npy"):
            partition = DataPartitioner(
                dataset.train,
                'load',
                self.n_client,
                {'show': False, 'alpha': self.args.alpha, 'batch_size': 64, 'num_workers': 2, 'load': f"./temp/{self.dataset_name}-{repeat}.npy"}
            )
        else:
            partition = DataPartitioner(
                dataset.train,
                self.args.partition,
                self.n_client,
                {'show': False, 'alpha': self.args.alpha, 'batch_size': 64, 'num_workers': 2, 'save': f"./temp/{self.dataset_name}-{repeat}.npy"}
            )

        testloader = DataLoader(dataset.test, batch_size=64)
            
        
        # set up the arguments for the homomorphic encryption
        pk, sk = PaillierKeypair.generate_keypair(1024)    
        init_model = model_select(self.model_name, num_classes=num_classes[self.dataset_name]).to(device)
        if not hasattr(self, 'clip'):
            self.clip = {key: 1 for key in init_model.state_dict().keys()}
        w_init = {key: value.flatten() for key, value in init_model.state_dict().items()}
        for base in basis:
            FedClient.count = 0
            logger.debug(f"Start the experiment at base '{base}'")
            # create models and initialize loacl models with global model
            local_models = [model_select(self.model_name, num_classes=num_classes[self.dataset_name]).to(device) for _ in range(self.n_client)]
            for model in local_models:
                model.load_state_dict(init_model.state_dict())
                
            dp_args = [{
                'n_aggr': self.n_round,
                'n_dataset': len(partition[i].dataset),
                'epsilon': self.epsilon,
                'delta': self.delta,
                'clip_thr': self.clip,
            } for i in range(self.n_client)]
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
                
            time_start = time.time()
            init_idcs = {}
            for key, value in w_init.items():
                init_idcs[key] = torch.randperm(len(value))[:math.ceil(rate * len(value))]
            # print(init_idcs)
            loss, acc = fed_train(clients, self.n_round, rate, init_idcs, partition.freq, vote=base)
            time_rec.append(time.time() - time_start)
            loss_rec.append(loss), acc_rec.append(acc)
        return loss_rec, acc_rec, time_rec
    
    def run_by_dists(self, dists: list, rate: float):
        loss_rec, acc_rec, time_rec = [], [], []
        dataset = Datasets(self.dataset_name)
        pk, sk = PaillierKeypair.generate_keypair(1024)    
        init_model = model_select(self.model_name, num_classes=num_classes[self.dataset_name]).to(device)
        if not hasattr(self, 'clip'):
            self.clip = {key: 1.2 for key in init_model.state_dict().keys()}
        w_init = {key: value.flatten() for key, value in init_model.state_dict().items()}
        testloader = DataLoader(dataset.test, batch_size=64)
        for dist in dists:
            FedClient.count = 0
            logger.debug(f"Start the experiment at distribution '{dist}'")
            if dist == 'iid':
                partition = DataPartitioner(
                    dataset.train,
                    'iid',
                    self.n_client,
                    {'show': True, 'batch_size': 64, 'num_workers': 2}
                )
            elif dist == 'pathetic':
                partition = DataPartitioner(
                    dataset.train,
                    'pathetic',
                    self.n_client,
                    {'show': True, 'batch_size': 64, 'num_workers': 2}
                )
            else:
                partition = DataPartitioner(
                    dataset.train,
                    'dirichlet',
                    self.n_client,
                    {'show': True, 'alpha': float(dist), 'batch_size': 64, 'num_workers': 2}
                )
                
            # create models and initialize loacl models with global model
            local_models = [model_select(self.model_name, num_classes=num_classes[self.dataset_name]).to(device) for _ in range(self.n_client)]
            for model in local_models:
                model.load_state_dict(init_model.state_dict())
                
            dp_args = [{
                'n_aggr': self.n_round,
                'n_dataset': len(partition[i].dataset),
                'epsilon': self.epsilon,
                'delta': self.delta,
                'clip_thr': self.clip,
            } for i in range(self.n_client)]
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
                
            time_start = time.time()
            init_idcs = {}
            for key, value in w_init.items():
                init_idcs[key] = torch.randperm(len(value))[:math.ceil(rate * len(value))]
            # print(init_idcs)
            loss, acc = fed_train(clients, self.n_round, rate, init_idcs, partition.freq)
            time_rec.append(time.time() - time_start)
            loss_rec.append(loss), acc_rec.append(acc)
        return loss_rec, acc_rec, time_rec

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
            elif vote != "global":
                idcs_glob = {}
                update = clients[0].get_raw_update()
                for key, value in update.items():
                    idcs_glob[key] = torch.randperm(len(value))[:math.ceil(he_rate * len(value))]
            # apply voting results to the clients
            for client in clients:
                if vote != "global":
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
                client.apply_updates(updates_glob, replace_idcs=(vote=="global"), he_frac=he_rate)
            
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