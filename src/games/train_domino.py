import argparse
import torch
import sys
import json
from domino import AlphaZeroTrainer, alpha_zero_net as azn

def main():
    # parser = argparse.ArgumentParser("Alpha Zero Trainer")

    # # Trainer init args
    # parser.add_argument('-bs', '--batch_size',    dest='batch_size',    type=int,   default=2048)
    # parser.add_argument('-h',  '--handouts',      dest='handouts',      type=int,   default=10)
    # parser.add_argument('-r',  '--rollouts',      dest='rollouts',      type=int,   default=10)
    # parser.add_argument('-tau',  '--tau_threshold', dest='tau_threshold', type=int,   default=6)
    # parser.add_argument('-n',  '--nine',          dest='pieces',        action='store_const', const=[9,10], default=[])
    # parser.add_argument('-d',  '--data_path',     dest='data_path',     default='domino/training/data')
    # parser.add_argument('-save',  '--save_path',     dest='save_path',     default='domino/training')

    # # Train params
    # parser.add_argument('-e',       '--epochs',             dest='epochs',          type=int,         default=100000)
    # parser.add_argument('-sim',     '--simulate',           dest='simulate',        type=bool,        default=True)
    # parser.add_argument('-l',       '--load_checkpoint',    dest='load_checkpoint', type=bool,        default=False)
    # parser.add_argument('-t',       '--tag',                dest='tag',             default='latest')
    # parser.add_argument('-np',      '--num_process',        dest='num_process',     type=int,         default=1)
    # parser.add_argument('-v',       '--verbose',            dest='verbose',         default='False')
    # parser.add_argument('-saveD',   '--save_data',          dest='save_data',       default='domino/training')

    # args = parser.parse_args()

    # Load configuration
    with open(sys.argv[1], "r") as reader:
        config = json.load(reader)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = azn(device=device)

    trainer = AlphaZeroTrainer(
        net,
        config['batch_size'],
        config['handouts'],
        config['rollouts'],
        config['max_number'],
        config['pieces_per_player'],
        config['data_path'],
        config['save_path'],
        config['tau_threshold'],
    )

    trainer.train(
        config['epochs'],
        config['simulate'],
        config['load_checkpoint'],
        config['tag'],
        config['num_process'],
        config['verbose'],
        config['save_data']
    )

if __name__ == '__main__':
    main()