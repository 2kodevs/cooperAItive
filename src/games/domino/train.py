import sys, json
from module import AlphaZeroTrainer

def main():
    # Load configuration
    with open(sys.argv[1], "r") as reader:
        config = json.load(reader)

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
        config['cput'],
        config['coop'],
        config['residual_layers'],
        config['num_filters'],
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
        config['num_gpus'],
        config['verbose'],
        config['save_data']
    )

if __name__ == '__main__':
    main()