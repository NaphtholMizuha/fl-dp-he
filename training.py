import math
from colorama import Fore, Style
import numpy as np
import torch
from frameworks import FedClient, FedServer
from utils import get_top_k_idcs


def aggregate_he_idcs(he_idcs, he_frac):
    idcs_locals = {}
    idcs_glob = {}
    # for key, value in he_idcs[0].items():
    #     print(f'{Fore.LIGHTRED_EX}key: {key}, k: {value.shape}{Style.RESET_ALL}')
    for he_idx in he_idcs:
        for key, value in he_idx.items():
            # print(Fore.LIGHTRED_EX, value, Style.RESET_ALL)
            if key in idcs_locals.keys():
                idcs_locals[key] = torch.cat((idcs_locals[key], value))
            else:
                idcs_locals[key] = value
                
    for key, value in idcs_locals.items():
        counter = torch.bincount(value)
        idcs_glob[key] = get_top_k_idcs(counter, len(he_idcs[0][key]))
        # print(f'{Fore.LIGHTRED_EX}key: {key}, k: {idcs_glob[key].shape}{Style.RESET_ALL}')
    return idcs_glob

def aggregate_paillier_dp(updates, freq):
    updates_glob = {}
    for id, he_update in enumerate(updates):
        for key in he_update.keys():
            if key in updates_glob.keys():
                updates_glob[key] += he_update[key] * freq[id]
            else:
                updates_glob[key] = he_update[key] * freq[id]
                
    return updates_glob
            
        
def paillier_dp_training(clients: list[FedClient], n_round: int, he_frac: float, init_he_idcs: dict, freq: dict):
    loss_li, acc_li = [], []
    
    for client in clients:
        client.protector.he_idcs = init_he_idcs
    
    for r in range(n_round):
        he_updates, dp_updates, top_idcs = [], [], []
        for client in clients:
            client.train_update()
            update = client.get_raw_update()
            he_update, dp_update = client.get_update()
            
            tops = {}
            for key, value in update.items():
                tops[key] = get_top_k_idcs(value, math.ceil(he_frac * len(value)))
                
            he_updates.append(he_update), dp_updates.append(dp_update), top_idcs.append(tops)
            
        he_glob = aggregate_paillier_dp(he_updates, freq)
        dp_glob = aggregate_paillier_dp(dp_updates, freq)
        idcs_glob = aggregate_he_idcs(top_idcs, he_frac)
        
        for client in clients:
            client.apply_updates((he_glob, dp_glob))
            client.protector.he_idcs = idcs_glob
            
    
        loss, acc = clients[0].log_test(r)
        loss_li.append(loss), acc_li.append(acc)
        
    return loss_li, acc_li
                
def he_dp_global_training(clients: list[FedClient], n_round: int, he_frac: float, init_he_idcs: dict, freq: dict):
    loss_li, acc_li = [], []
    
    for client in clients:
        client.protector.he_idcs = init_he_idcs
    
    for r in range(n_round):
        he_updates, dp_updates = [], []
        for client in clients:
            client.train_update()
            he_update, dp_update = client.get_update()
            he_updates.append(he_update), dp_updates.append(dp_update)
            
        he_glob = aggregate_paillier_dp(he_updates, freq)
        dp_glob = aggregate_paillier_dp(dp_updates, freq)
        
        for client in clients:
            client.apply_updates((he_glob, dp_glob), replace_idcs=True, he_frac=he_frac)
            
        loss, acc = clients[0].log_test(r)
        loss_li.append(loss), acc_li.append(acc)
        
    return loss_li, acc_li              
            
def he_dp_random_training(clients: list[FedClient], n_round: int, he_frac: float, init_he_idcs: dict, freq: dict):
    loss_li, acc_li = [], []
    
    for client in clients:
        client.protector.he_idcs = init_he_idcs
    
    for r in range(n_round):
        he_updates, dp_updates = [], []
        for client in clients:
            client.train_update()
            he_update, dp_update = client.get_update()
            he_updates.append(he_update), dp_updates.append(dp_update)
            
        he_glob = aggregate_paillier_dp(he_updates, freq)
        dp_glob = aggregate_paillier_dp(dp_updates, freq)
        next_he_idcs = {}
        for key, value in clients[0].get_raw_update().items():
            next_he_idcs[key] = np.random.choice(np.arange(0, len(value)), size=math.ceil(he_frac * len(value)), replace=False)
        for client in clients:
            client.apply_updates((he_glob, dp_glob))
            
        loss, acc = clients[0].log_test(r)
        loss_li.append(loss), acc_li.append(acc)
        
    return loss_li, acc_li             
            
    