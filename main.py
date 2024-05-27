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

def exp_clips(args):
    exp = Experiment(args)
    clips = [0.1, 0.25, 0.5, 0.8, 1]
    if not os.path.exists(f"./results/clips-{args.data}-{args.model}-{args.exp}"):
        os.makedirs(f"./results/clips-{args.data}-{args.model}-{args.exp}")
    for i in range(args.repeat):
        loss, acc, time = exp.run_by_clips(clips, args.repeat)
    
        loss_df = pd.DataFrame(loss, index=clips).transpose()
        acc_df = pd.DataFrame(acc, index=clips).transpose()
        time_df = pd.DataFrame(time, index=clips).transpose()
    
        loss_df.to_csv(f"./results/{args.exp}-{args.data}-{args.model}-loss{i}.csv")
        acc_df.to_csv(f"./results/{args.exp}-{args.data}-{args.model}-acc{i}.csv")
        time_df.to_csv(f"./results/{args.exp}-{args.data}-{args.model}-time{i}.csv")

def exp_rates(args):
    exp = Experiment(args)
    
    rates = [0, 0.1, 0.25, 0.5, 0.8, 1]
    if not os.path.exists(f"./results/rates-{args.data}-{args.model}-{args.exp}"):
        os.makedirs(f"./results/rates-{args.data}-{args.model}-{args.exp}")
    for i in range(args.repeat):
        loss, acc, time = exp.run_by_rates(rates)
    
        loss_df = pd.DataFrame(loss, index=rates).transpose()
        acc_df = pd.DataFrame(acc, index=rates).transpose()
        time_df = pd.DataFrame(time, index=rates).transpose()
    
        loss_df.to_csv(f"./results/{args.data}-{args.model}-{args.exp}/loss{i}.csv")
        acc_df.to_csv(f"./results/{args.data}-{args.model}-{args.exp}/acc{i}.csv")
        time_df.to_csv(f"./results/{args.data}-{args.model}-{args.exp}/time{i}.csv")

def exp_rates_v(args):
    
    rates = [0, 0.1, 0.25, 0.5, 0.8, 1]
    eps = [0.05, 0.1, 0.25, 0.5, 1]
    
    
    for e in eps:
        args.epsilon = e
        exp = Experiment(args)

        for i in range(args.repeat):
            loss, acc, time = exp.run_by_rates(rates, i)
        
            loss_df = pd.DataFrame(loss, index=rates).transpose()
            acc_df = pd.DataFrame(acc, index=rates).transpose()
            time_df = pd.DataFrame(time, index=rates).transpose()
        
            loss_df.to_csv(f"./results/{args.exp}-{e}-{args.data}-{args.model}-loss{i}.csv")
            acc_df.to_csv(f"./results/{args.exp}-{e}-{args.data}-{args.model}-acc{i}.csv")
            time_df.to_csv(f"./results/{args.exp}-{e}-{args.data}-{args.model}-time{i}.csv")

    
def exp_basis(args):
    exp = Experiment(args)
    basis = ["global", "random", "best", "worst"]
    if not os.path.exists(f"./results/basis-{args.data}-{args.model}-{args.exp}"):
        os.makedirs(f"./results/basis-{args.data}-{args.model}-{args.exp}")
    for i in range(args.repeat):
        loss, acc, time = exp.run_by_basis(basis, 0.1)
    
        loss_df = pd.DataFrame(loss, index=basis).transpose()
        acc_df = pd.DataFrame(acc, index=basis).transpose()
        time_df = pd.DataFrame(time, index=basis).transpose()
    
        loss_df.to_csv(f"./results/{args.data}-{args.model}-{args.exp}/loss{i}.csv")
        acc_df.to_csv(f"./results/{args.data}-{args.model}-{args.exp}/acc{i}.csv")
        time_df.to_csv(f"./results/{args.data}-{args.model}-{args.exp}/time{i}.csv")
        
def exp_basis_v(args):
    basis = ["global", "random", "best", "worst"]
    eps = [0.05, 0.1, 0.25, 0.5, 1]
    
    for e in eps:
        args.epsilon = e
        exp = Experiment(args)
        for i in range(args.repeat):
            loss, acc, time = exp.run_by_basis(basis, 0.1, i)
        
            loss_df = pd.DataFrame(loss, index=basis).transpose()
            acc_df = pd.DataFrame(acc, index=basis).transpose()
            time_df = pd.DataFrame(time, index=basis).transpose()
        
            loss_df.to_csv(f"./results/{args.exp}-{e}-{args.data}-{args.model}-loss{i}.csv")
            acc_df.to_csv(f"./results/{args.exp}-{e}-{args.data}-{args.model}-acc{i}.csv")
            time_df.to_csv(f"./results/{args.exp}-{e}-{args.data}-{args.model}-time{i}.csv")

def exp_dists(args):
    exp = Experiment(args)
    dists = ["pathetic", "iid", "1.0", "0.5", "0.1"]
    if not os.path.exists(f"./results/dists-{args.data}-{args.model}-{args.exp}"):
        os.makedirs(f"./results/dists-{args.data}-{args.model}-{args.exp}")
    for i in range(args.repeat):
        loss, acc, time = exp.run_by_dists(dists, 0.1)
    
        loss_df = pd.DataFrame(loss, index=dists).transpose()
        acc_df = pd.DataFrame(acc, index=dists).transpose()
        time_df = pd.DataFrame(time, index=dists).transpose()
    
        loss_df.to_csv(f"./results/dists-{args.data}-{args.model}-{args.exp}/loss{i}.csv")
        acc_df.to_csv(f"./results/dists-{args.data}-{args.model}-{args.exp}/acc{i}.csv")
        time_df.to_csv(f"./results/dists-{args.data}-{args.model}-{args.exp}/time{i}.csv")
    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Experiment for DP-HE-FedAvg")
    parser.add_argument('--exp', type=str, default='rates', help='Experiment type')
    parser.add_argument('--data', type=str, default='cifar10', help='Dataset name')
    parser.add_argument('--model', type=str, default='lenet5', help='Model name')
    parser.add_argument('--epochs', type=int, default=3, help='Local epochs')
    parser.add_argument('--rounds', type=int, default=30, help='Global rounds')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--clients', type=int, default=10, help='Number of clients')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Privacy budget')
    parser.add_argument('--delta', type=float, default=1e-5, help='Privacy budget')
    parser.add_argument('--log_client', type=bool, default=False, help='Log client')
    parser.add_argument('--test_client', type=bool, default=False, help='Test client')
    parser.add_argument('--partition', type=str, default="dirichlet", help='Data partition')
    parser.add_argument('--alpha', type=float, default=0.5, help='Dirichlet alpha')
    parser.add_argument('--repeat', type=int, default=5, help='Repeat times')
    parser.add_argument('--vareps', type=bool, default=False, help='Varying epsilon')
    args = parser.parse_args()
    logger.add(f"./results/{args.data}-{args.model}-{args.exp}.log", format="{time:YYYY-MM-DD HH:mm:ss} {level} {message}")
    if not args.vareps:
        if args.exp == 'rates':
            exp_rates(args)
        elif args.exp == 'basis':
            exp_basis(args)
        elif args.exp == 'dists':
            exp_dists(args)
        elif args.exp == 'clips':
            exp_clips(args)
    else:
        if args.exp == 'rates':
            exp_rates_v(args)
        elif args.exp == 'basis':
            exp_basis_v(args)
        elif args.exp == 'dists':
            raise NotImplementedError