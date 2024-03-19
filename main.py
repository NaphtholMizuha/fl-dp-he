import os
from loguru import logger
from experiment import experiment
import pickle
import datetime
device = 'cuda'
os.environ["CUDA_VISIBLE_DEVICES"]= "0,1"
logger.add(f"./results/{datetime.datetime.now()}.log", format="{time:YYYY-MM-DD HH:mm:ss} {level} {message}")

if __name__ == "__main__":
    results = experiment("cifar10", "simplecnn")
    print(results)
    pickle.dump(results, open("./results/results.pkl", "wb"))