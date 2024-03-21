import os
from loguru import logger
from experiment import Experiment
import pickle
import datetime
from utils import plot_contrast
import pandas as pd
import argparse
time_start = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
device = 'cuda'
os.environ["CUDA_VISIBLE_DEVICES"]= "0,1"


def exp_rates(dataset, model):
    exp = Experiment(dataset_name=dataset, model_name=model)
    rates = [0, 0.1, 0.25, 0.5, 0.8, 1]
    acc, loss = exp.run_by_rates(rates)
    plot_contrast("accuracy", f"rates-{dataset}-{model}-", acc, rates)
    plot_contrast("loss", f"rates-{dataset}-{model}-", loss, rates)
    
    loss_min = {rate: min(loss[i]) for rate, i in enumerate(loss)}
    acc_max = {rate: max(acc[i]) for rate, i in enumerate(acc)}
    
    loss_df = pd.DataFrame(loss_min)
    acc_df = pd.DataFrame(acc_max)
    
    loss_df.to_csv(f"./rates-{dataset}-{model}.csv")
    acc_df.to_csv(f"./rates-{dataset}-{model}.csv")
    
def exp_basis(dataset, model):
    exp = Experiment(dataset_name=dataset, model_name=model)
    basis = ["random", "best", "worst"]
    acc, loss = exp.run_by_basis(basis, 0.25)
    plot_contrast("accuracy", f"basis-{dataset}-{model}-", acc, basis)
    plot_contrast("loss", f"basis-{dataset}-{model}-", loss, basis)
    
    loss_min = {rate: min(loss[i]) for rate, i in enumerate(loss)}
    acc_max = {rate: max(acc[i]) for rate, i in enumerate(acc)}
    
    loss_df = pd.DataFrame(loss_min)
    acc_df = pd.DataFrame(acc_max)
    
    loss_df.to_csv(f"./basis-{dataset}-{model}.csv")
    acc_df.to_csv(f"./basis-{dataset}-{model}.csv")
    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Experiment for DP-HE-FedAvg")
    parser.add_argument('-e', '--exp', type=str, default='rates', help='Experiment type')
    parser.add_argument('-d', '--data', type=str, default='cifar10', help='Dataset name')
    parser.add_argument('-m', '--model', type=str, default='lenet5', help='Model name')
    args = parser.parse_args()
    logger.add(f"./results/{args.data}-{args.model}-{args.exp}.log", format="{time:YYYY-MM-DD HH:mm:ss} {level} {message}")
    if args.exp == 'rates':
        exp_rates(args.data, args.model)
    elif args.exp == 'basis':
        exp_basis(args.data, args.model)