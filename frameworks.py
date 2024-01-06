import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loguru import logger
from aggregator import Aggregator
from privacy import BaseDp
from trainer import Trainer
import uuid

device = "cuda"

# TODO: How to apply the update splited by HE and DP?

class FedServer:
    def __init__(
        self,
        aggregator: Aggregator,
        freq: dict,
        device: str = 'cuda',
        weight = dict,
        args: dict = {},
    ):
        self.uuid = uuid.uuid1()
        self.aggregator = aggregator
        self.freq = freq
        self.args = args
        self.weight = weight
        self.device = device

    def aggregate(self, weights):
        self.aggregator.aggregate(weights, self.freq)
        
    def aggregate_updates(self, updates):
        update = self.aggregator.aggregate_update(updates, self.freq)
        for key in update.keys():
            self.weight[key] += update[key]
        return self.weight

class FedClient:
    count = 0
    def __init__(
        self,
        model: nn.Module,
        trainer: Trainer,
        args: dict = {},
        testloader: DataLoader = None,
        protector: BaseDp = None
    ):
        self.uuid = uuid.uuid1()
        self.id = FedClient.count
        FedClient.count += 1
        self.trainer = trainer
        self.args = args
        self.testloader = testloader
        self.model = model
        self.protector = protector



    def train(self):
        for i in range(self.args['epoch']):
            loss = self.trainer.train()
            if self.args['log']:
                logger.debug(f"Client {self.id} Epoch {i} : train_loss={loss:.3f}")
        if self.args['test']:
                loss, acc = self.test()
                logger.info(f"Client {self.id}: test_loss={loss:.3f}, acc={acc*100:.2f}%")
        # for k, v in self.model.state_dict().items():
        #     print(f"'{k}': {torch.norm(v)}")
        
    def train_update(self):
        older = copy.deepcopy(self.model.state_dict())
        self.train()
        newer = self.model.state_dict()
        self.prev = older
        self.update = {key: newer[key] - older[key] for key in older.keys()}

    def get_weights(self):
        weight = {key: value.flatten() for key, value in self.model.state_dict().items()}
        if self.protector is None:
            return weight
        else:
            return self.protector(weight)

    def set_weights(self, weights: dict):
        if self.protector is not None and hasattr(self.protector, 'recover'):
            weights = self.protector.recover(weights)
        weights = {key: value.reshape(self.model.state_dict()[key].shape) for key, value in weights.items()}
        self.model.load_state_dict(weights)

    def get_update(self):
        update = {key: value.flatten() for key, value in self.update.items()}
        if self.protector is None:
            return update
        else:
            return self.protector(update)
        
        
    def log_test(self, round):
        loss, acc = self.test()
        logger.success(f"Server Round {round}: loss={loss:.3f}, acc={acc * 100:.2f}%")
        return loss, acc

    def test(self):
        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        loss, acc = 0, 0

        with torch.no_grad():
            for x, y in self.testloader:
                x, y = x.to(device), y.to(device)
                pred = self.model(x)
                loss += criterion(pred, y).item()
                acc += (pred.argmax(1) == y).type(torch.float).sum().item()

        loss /= len(self.testloader)
        acc /= len(self.testloader.dataset)
        return loss, acc

    def __del__(self):
        self.count -= 1