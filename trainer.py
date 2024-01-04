import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
device = "cuda"

class Trainer:
    def __init__(self, model: nn.Module, dataloader: DataLoader, criterion, args={}):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.args = args

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr = self.args['lr'],
            momentum=0.9
        )

    def train(self):
        self.model.train()
        loss_sum = 0.0

        for x, y in self.dataloader:
            x, y = x.to(device), y.to(device)
            pred = self.model(x)
            loss = self.criterion(pred, y)
            loss.backward()
            loss_sum += loss.item()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss_sum / len(self.dataloader)

    def __call__(self):
        self.train()

class ExpectationBasedTrainer(Trainer):
    def __init__(self, model: nn.Module, dataloader: DataLoader, criterion, args={}):
        super().__init__(model, dataloader, criterion, args)
        self.std = args['std']

    def train(self):
        self.model.train()
        loss_sum = 0.0

        for x, y in self.dataloader:
            x, y = x.to(device), y.to(device)
            pred = self.model(x)
            loss = self.criterion(pred, y)
            grad_params = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
            grad_norm = 0.0
            for grad in grad_params:
                grad_norm += torch.norm(grad) ** 2
            loss += grad_norm * (self.std ** 2)
            loss.backward()
            loss_sum += loss.item()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss_sum / len(self.dataloader)