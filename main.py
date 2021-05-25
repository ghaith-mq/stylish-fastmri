import sys
import pathlib as pb
import argparse as ap
import random

import yaml
import torch
import numpy as np

ROOT_PATH = pb.Path(__file__).resolve().parent
sys.path.append(str(ROOT_PATH))
from trainer import trainer_default, trainer_default_base


random.seed(0)
np.random.seed(0)
torch.use_deterministic_algorithms(True)
torch.manual_seed(0)




def build_args():
    parser = ap.ArgumentParser(description='Main entry to train/evaluate models.')
    parser.add_argument('config', nargs='+', type=str)
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='GPU or CPU')
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--mode', type=str, choices=['train', 'val', 'test'], default='train')
    parser.add_argument('--dataset-path', type=str, default=None)
    parser.add_argument('--logs-dir', type=str, default=None)
    parser.add_argument('--train-dataset-cache-file', type=str, default=None)
    parser.add_argument('--val-dataset-cache-file', type=str, default=None)
    parser.add_argument('--test-dataset-cache-file', type=str, default=None)
    args = parser.parse_args()
    
    config = {}
    for c in args.config:
        with open(c, 'r') as f:
            config_ = yaml.safe_load(f)
        config.update(config_)
        
    for k, v in config.items():
        if not hasattr(args, k) or getattr(args, k) is None:
            setattr(args, k, v)
        
    return args


def main():
    args = build_args()
    
    if args.trainer == 'default':
        trainer = trainer_default.FastMRIDefaultTrainer(**vars(args))
    elif args.trainer == 'default_base':
        trainer = trainer_default_base.FastMRIDefaultBaseTrainer(**vars(args))
    else:
        raise NotImplementedError()
        
    if args.mode == 'train':
        trainer.train(args.epochs)
    else:
        raise NotImplementedError()

if __name__ == '__main__':
    main()
