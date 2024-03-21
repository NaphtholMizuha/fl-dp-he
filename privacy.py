import math
import tenseal as ts
from abc import ABC, abstractmethod
from math import sqrt
import contextlib
import torch
from utils import SparseVector
import io
device = 'cuda'

class BaseProtector(ABC):
    @abstractmethod
    def protect(self, data):
        pass

    @abstractmethod
    def __call__(self, weight):
        pass

class BaseDp():
    def clip(self, weight):
        clipped = {}
        for key, value in weight.items():
            if "weight" not in key and "bias" not in key:
                clipped[key] = value
                continue
            clipped[key] = value / max(torch.norm(value) / self.clip_thr[key], 1)
        return clipped

    def protect(self, weight):
        weight = self.clip(weight)
        for key in weight.keys():
            shape = weight[key].shape
            if "weight" not in key and "bias" not in key:
                continue
            weight[key] += torch.randn(shape).to(device) * self.std[key]
        return weight

    def __call__(self, weight):
        return self.protect(weight)

class VanillaDp(BaseDp):
    def __init__(self, args: dict={}):
        self.n_aggr = args['n_aggr']
        self.n_dataset = args['n_dataset']
        self.epsilon = args['epsilon']
        self.delta = args['delta']
        self.clip_thr = args['clip_thr']
        self.std = self.get_std()

    def get_std(self):
        std = {}
        for key, value in self.clip_thr.items():
            sensitivity = 2 * value / self.n_dataset
            std[key] = sensitivity * sqrt(2 * self.n_aggr * math.log(1/self.delta)) / self.epsilon
        return std

class RenyiDp(BaseDp):
    def __init__(self, args: dict={}):
        self.n_aggr = args['n_aggr']
        self.n_dataset = args['n_dataset']
        self.alpha = args['alpha']
        self.epsilon = args['epsilon']
        self.clip_thr = args['clip_thr']
        self.std = self.get_std()

    def get_std(self):
        std = {}
        for key, value in self.clip_thr.items():
            sensitivity = 2 * value / self.n_dataset
            std[key] = sensitivity * sqrt(self.alpha / (2 * self.epsilon))
        return std

class Ckks():
    def __init__(self, ctx: ts.Context):
        self.ctx = ctx
        
    def ckks_to_torch(cipher: ts.CKKSTensor): 
        return cipher.decrypt()
        
    def protect(self, data):
        with contextlib.redirect_stdout(io.StringIO()):
            enc = {key: ts.ckks_vector(self.ctx, value.cpu()) for key, value in data.items()}
        return enc
    
    def recover(self, data):
        return {key: torch.tensor(value.decrypt()).to(device) for key, value in data.items()}
    
    def __call__(self, weight):
        return self.protect(weight)

class Paillier():
    def __init__(self, pk, sk):
        self.pk = pk
        self.sk = sk

    def protect(self, data):
        encrypted_data = {key: self.pk.encrypt(value.cpu().numpy()) for key, value in data.items()}
        return encrypted_data

    def recover(self, data):
        decrypted_data = {key: torch.tensor(self.sk.decrypt(value)).to(device) for key, value in data.items()}
        return decrypted_data

    def __call__(self, weight):
        return self.protect(weight)
    
class PaillierDp:
    def __init__(self, pk, sk, he_idcs: dict, args: dict={}) -> None:
        self.paillier = Paillier(pk, sk)
        self.dp = VanillaDp(args)
        self.he_idcs = he_idcs

    def split(self, data: dict[str, torch.Tensor]):
        he_data, dp_data = {}, {}
        for key, value in data.items():
            he_data[key] = value[self.he_idcs[key]]
            dp_data[key] = value.to(device)
        return he_data, dp_data
    
    def recover(self, data):
        he_data, dp_data = data
        he_data = self.paillier.recover(he_data)
        data = {}
        for key in he_data.keys():
            data[key] = dp_data[key]
            data[key][self.he_idcs[key]] = he_data[key].to(device)
        return data
    
    def protect(self, data: dict[str, torch.Tensor]):
        he_data, dp_data = self.split(data)
        he_data = self.paillier(he_data)
        dp_data = self.dp(dp_data)
        return he_data, dp_data
    
    def __call__(self, weight):
        return self.protect(weight)

class CkksDp:
    def __init__(self, context: ts.Context, he_idcs: dict, args: dict) -> None:
        self.ckks = Ckks(context)
        self.dp = VanillaDp(args)
        self.he_idcs = he_idcs

    def split(self, data: dict[str, torch.Tensor]):
        ckks_data, dp_data = {}, {}
        for key, value in data.items():
            ckks_data[key] = value[self.he_idcs[key]]
            dp_data[key] = value.to(device)
            dp_data[key][self.he_idcs[key]] = 0
        return ckks_data, dp_data
    
    def recover(self, data):
        he_data, dp_data = data
        he_data = self.ckks.recover(he_data)
        data = {}
        for key in he_data.keys():
            data[key] = dp_data[key]
            data[key][self.he_idcs[key]] = he_data[key].to(device)
        return data
    
    def protect(self, data: dict[str, torch.Tensor]):
        ckks_data, dp_data = self.split(data)
        ckks_data = self.ckks(ckks_data)
        dp_data = self.dp(dp_data)
        return ckks_data, dp_data
    
    def __call__(self, weight):
        return self.protect(weight)
