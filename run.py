import subprocess

# subprocess.run('screen -dm bash -c "python3 script1.py"', shell=True)

cmds = [
    # "/home/wuzihou/miniconda3/envs/FL/bin/python main.py --exp=rates --data=mnist --model=2nn --repeat=5 --vareps=True",
    # "/home/wuzihou/miniconda3/envs/FL/bin/python main.py --exp=rates --data=fmnist --model=cnn --repeat=5 --vareps=True",
    # "/home/wuzihou/miniconda3/envs/FL/bin/python main.py --exp=rates --data=cifar10 --model=lenet5 --repeat=5 --vareps=True",
    # "/home/wuzihou/miniconda3/envs/FL/bin/python main.py --exp=rates --data=cifar100 --model=resnet9 --repeat=5 --vareps=True",
    
    # "/home/wuzihou/miniconda3/envs/FL/bin/python main.py --exp=basis --data=mnist --model=2nn --repeat=5 --vareps=True",
    # "/home/wuzihou/miniconda3/envs/FL/bin/python main.py --exp=basis --data=fmnist --model=cnn --repeat=5 --vareps=True",
    # "/home/wuzihou/miniconda3/envs/FL/bin/python main.py --exp=basis --data=cifar10 --model=lenet5 --repeat=5 --vareps=True",
    # "/home/wuzihou/miniconda3/envs/FL/bin/python main.py --exp=basis --data=cifar100 --model=resnet9 --repeat=5 --vareps=True",
    
    # "/home/wuzihou/miniconda3/envs/FL/bin/python main.py --exp=dists --data=mnist --model=2nn --repeat=5",
    # "/home/wuzihou/miniconda3/envs/FL/bin/python main.py --exp=dists --data=fmnist --model=cnn --repeat=5",
    # "/home/wuzihou/miniconda3/envs/FL/bin/python main.py --exp=dists --data=cifar10 --model=lenet5 --repeat=5",
    # "/home/wuzihou/miniconda3/envs/FL/bin/python main.py --exp=dists --data=cifar100 --model=resnet9 --repeat=5",
    
    "/home/wuzihou/miniconda3/envs/FL/bin/python main.py --exp=clips --data=mnist --model=2nn --repeat=5",
    "/home/wuzihou/miniconda3/envs/FL/bin/python main.py --exp=clips --data=fmnist --model=cnn --repeat=5",
    "/home/wuzihou/miniconda3/envs/FL/bin/python main.py --exp=clips --data=cifar10 --model=lenet5 --repeat=5",
    "/home/wuzihou/miniconda3/envs/FL/bin/python main.py --exp=clips --data=cifar100 --model=resnet9 --repeat=clips",
]

names = [
    #'mnist-rates', 'fmnist-rates', 'cifar-rates', 'resnet-rates',
    # 'mnist-basis', 'fmnist-basis', 'cifar-basis', 'resnet-basis',
    # "mnist-dists", "fmnist-dists", "cifar10-dists", "cifar100-dists",
    "mnist-clips", "fmnist-clips", "cifar10-clips", "cifar100-clips",
]
for cmd, name in zip(cmds, names):
    subprocess.run(f'screen -dmS {name} bash -c "{cmd}"', shell=True)