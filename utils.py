from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch

class SparseVector:
    def __init__(self, data):
        self.data = {i: val for i, val in enumerate(data) if val != 0}
        
    @classmethod
    def from_dict(cls, data_dict):
        new = cls([])
        new.data = data_dict
        return new
    
    @classmethod
    def from_idcs(cls, idcs, data):
        new = cls([])
        new.data = {i: data[i] for i in idcs}
        return new

    def __iter__(self):
        self.iter = iter(self.data.items())
        return self

    def __next__(self):
        return next(self.iter)
    
    def __repr__(self) -> str:
        return str(self.data)

    def __add__(self, other):
        result = self.data.copy()
        for index, value in other.data.items():
            if index in result:
                result[index] += value
            else:
                result[index] = value
        return result


def plot_contrast(title, data, labels):
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    for i, datum in enumerate(data):
        plt.plot(datum, color=colors[i], label=labels[i], marker='.')
        plt.legend()
    plt.title(title)
    plt.savefig(f'./results/{title}.png')
    plt.close()

def bar_chart(title, data, labels, path='./results/'):
    plt.bar(labels, data)
    plt.title(title)
    plt.savefig(f'./results/{title}.png')
    plt.close()
    
def histogram(title, data):
    plt.hist(data, bins=20)
    plt.title(title)
    plt.savefig(f'./results/{title}.png')
    plt.close()
    
def top_k_idcs(data: torch.Tensor, k: int):
    return torch.argsort(data, descending=True)[:k]

def series_top_k_idcs(data: list[torch.Tensor], key, round):
    # make the list of tensors into a tensor
    data = torch.concatenate(data)

    counts = torch.bincount(data)
    counts_sorted, indices_sorted = torch.sort(counts, descending=True)
    bar_chart(f'Round {round}: {key}' ,counts_sorted.to('cpu'), indices_sorted.to('cpu'), path='./results/topk')