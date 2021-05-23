import sys
import pathlib as pb
import argparse as ap

import yaml

ROOT_PATH = pb.Path(__file__).resolve().parent
sys.path.append(str(ROOT_PATH))
from trainer import trainer_default



def build_args():
    parser = ap.ArgumentParser(description='Main entry to train/evaluate models.')
    parser.add_argument('config', nargs='+', type=str, required=True)
    parser.add_argument('--device', type=int, default='cuda', choices=['cpu', 'cuda'], help='GPU or CPU')
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--mode', type=str, choices=['train', 'val', 'test'], default='train')
    args = parser.parse_args()
    
    config = {}
    for c in args.config:
        with open(c, 'r') as f:
            config_ = yaml.safe_load(f)
        config.update(config_)
        
    for k, v in config.items():
        setattr(args, k, v)
        
    return args


def main():
    args = build_args()
    
    if args.trainer == 'default':
        trainer = trainer_default.FastMRIDefaultTrainer(vars(args))
    else:
        raise NotImplementedError()
        
    if args.mode == 'train':
        trainer.train(args.epochs)
    else:
        raise NotImplementedError()
    
if __name__ == '__main__':
    main()
