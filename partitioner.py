import torch.utils.data
from matplotlib.ticker import MaxNLocator
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import matplotlib.pyplot as plt

class DataPartitioner:
    def __init__(self, dataset: Dataset, partition, n_client, args={}):
        self.dataset = dataset
        self.labels = np.array([item[1] for item in self.dataset])
        self.partition = partition
        self.n_class = self.labels.max() + 1
        self.n_client = n_client
        self.args = args

        if self.partition == "iid":
            partition_dict = self.iid_partition()
        elif self.partition == "dirichlet":
            partition_dict = self.dirichlet_partition()

        self.freq = {i: len(partition_dict[i]) / len(self.dataset) for i in range(n_client)}
        self.dataloaders = self.split(partition_dict)

        if self.args['show'] :
            self.show_distribution(partition_dict)



    def iid_partition(self):
        indices = np.array([i for i in range(len(self.dataset))])
        np.random.shuffle(indices)
        indices = np.split(indices, self.n_client)

        return {i : indices[i] for i in range(self.n_client)}


    def dirichlet_partition(self):
        alpha = self.args['alpha']
        label_dist = np.random.dirichlet([alpha] * self.n_client, self.n_class)
        class_idcs = [np.argwhere(self.labels == y).flatten()
                      for y in range(self.n_class)]
        client_idcs = [[] for _ in range(self.n_client)]

        for k_idcs, fracs in zip(class_idcs, label_dist):
            for i, idcs in enumerate(
                np.split(
                    k_idcs,
                    (np.cumsum(fracs)[:-1] * len(k_idcs)).astype(int)
                )
            ):
                client_idcs[i] += [idcs]

        return {i: np.concatenate(idcs)
                for i, idcs in enumerate(client_idcs)}

    def split(self, partition_dict):
        subsets = {}
        for i, idcs in partition_dict.items():
            subsets[i] = Subset(self.dataset, idcs)

        return {i : DataLoader(
            subsets[i],
            batch_size=self.args['batch_size'],
            num_workers=self.args['num_workers'],
            shuffle=True
        ) for i in range(self.n_client)}

    def show_distribution(self, partition_dict: dict):
        mat = np.zeros([self.n_client, self.n_class])
        for client, indices in partition_dict.items():
            for idx in indices:
                mat[client][int(self.dataset.targets[idx])] += 1
        # print(mat)
        plt.matshow(mat, vmin=0, vmax=4000, cmap=plt.cm.Blues)
        plt.xlabel('Client')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.ylabel('Class')
        plt.title("Distribution")
        plt.colorbar()
        plt.savefig('./results/dist.png')
        plt.close()

    def __getitem__(self, item) -> DataLoader:
        return self.dataloaders[item]