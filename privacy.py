import math
import tenseal as ts
from abc import ABC, abstractmethod
from math import sqrt
import contextlib
import torch
from lightphe import LightPHE
import io
device = 'cpu'

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
            clipped[key] = value / max(torch.norm(value) / self.clip_thr[key], 1)
        return clipped

    def protect(self, weight):
        weight = self.clip(weight)
        for key in weight.keys():
            shape = weight[key].shape
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

class CkksHe():
    def __init__(self, ctx: ts.Context):
        self.ctx = ctx
        
    def ckks_to_torch(cipher: ts.CKKSTensor):
        plain = cipher.decrypt()
        return torch.tensor(plain.raw).reshape(plain.shape)
        
    def protect(self, data):
        with contextlib.redirect_stdout(io.StringIO()):
            enc = {key: ts.ckks_vector(self.ctx, value.cpu()) for key, value in data.items()}
        return enc
    
    def recover(self, data):
        return {key: torch.tensor(value.decrypt()).to(device) for key, value in data.items()}
    
    def __call__(self, weight):
        return self.protect(weight)

class PaillierHe():
    def __init__(self):
        self.cs = LightPHE(algorithm_name='Paillier')

    def protect(self, data):
        encrypted_data = {key: self.cs.encrypt(value.cpu().tolist()) for key, value in data.items()}
        return encrypted_data

    def recover(self, data):
        decrypted_data = {key: torch.tensor(self.cs.decrypt(value)).to(device) for key, value in data.items()}
        return decrypted_data

    def __call__(self, weight):
        return self.protect(weight)

class CkksDp:
    def __init__(self, context: ts.Context, he_indcs: dict, args: {}) -> None:
        self.ckks = CkksHe(context)
        self.dp = VanillaDp(args)
        self.he_indcs = he_indcs

    def split(self, data: dict[str, torch.Tensor]):
        ckks_data, dp_data = {}, {}
        for key, value in data.items():
            other_idcs = [i for i in range(value.shape[0]) if i not in self.he_indcs[key]]
            ckks_data[key] = value[self.he_indcs[key]]
            dp_data[key] = value[other_idcs]
        return ckks_data, dp_data
    
    def protect(self, data: dict[str, torch.Tensor]):
        ckks_data, dp_data = self.split(data)
        ckks_data = self.ckks(ckks_data)
        dp_data = self.dp(dp_data)
        return ckks_data, dp_data
    
    def __call__(self, weight):
        return self.protect(weight)
