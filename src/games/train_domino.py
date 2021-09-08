import torch
import sys
import json
from domino import AlphaZeroTrainer, alpha_zero_net as azn

def main():
    # Load configuration
    with open(sys.argv[1], "r") as reader:
        config = json.load(reader)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = azn(device=device)

    trainer = AlphaZeroTrainer(
        config['batch_size'],
        config['handouts'],
        config['rollouts'],
        config['alpha'],
        config['max_number'],
        config['pieces_per_player'],
        config['data_path'],
        config['save_path'],
        config['lr'],
        config['tau_threshold'],
    )

    trainer.train(
        config['epochs'],
        config['simulate'],
        config['load_checkpoint'],
        config['load_model'],
        config['sample'],
        config['tag'],
        config['num_process'],
        config['verbose'],
        config['save_data']
    )

if __name__ == '__main__':
    main()