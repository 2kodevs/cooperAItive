from multiprocessing import Pool
from ..models import alpha_zero_net as Net
from .trainer import Trainer

import numpy as np
import random
import time
import os
import json

class AlphaZeroTrainer(Trainer):
    """
    Trainer manager for Alpha Zero model
    """
    def __init__(self, game: Domino, net: Net, batch_size: int, rollouts: int, data_path: str):
        """
        param game:
            Manager of the game in which the agent is a player
        param net: nn.Module
            Neural Network to train
        param batch_size: int
            Size of training data used for epoch
        """
        self.game = game
        self.net = net
        self.batch_size = batch_size
        self.rollouts = rollouts
        self.data_path = data_path
        self.error_log = []

        self.net.eval()
        
    def self_play(self, rollouts):
        # //TODO: Rulo
        # Only one game simulated here, and save game data.
        # This method will be called by policy_iteration,
        # use all the params that you like, or properties of the class.
        # You can change init to accept more args.
        # The NN is in self.net, you can pass it to your methods.
        pass

    def policy_iteration(self, epoch: int, simulate: bool, num_process=1, verbose=False, save_data=False):
        data = self.get_data(simulate, num_process, verbose, self.batch_size)
        
        if save_data:
            num_files = len([name for name in os.listdir(self.data_path) if os.path.isfile(name)])
            path = f'{self.data_path}/training_data_{num_files}'
            with open(path, 'w') as writer:
                json.dump(data, writer)

        batch = random.sample(data, self.batch_size)
        loss = self.net.train(batch)
        Trainer.adjust_learning_rate(epoch, self.net.optimizer)
        self.error_log.append(loss)
        self.net.save(self.error_log, verbose=True)
        return loss

    def get_data(self, simulate: bool, num_process: int, verbose: bool, batch_size: int):
        data = []
        num_games = 0

        if simulate:
            if verbose:
                print('Simulations started')
                start = time.time()

            if num_process > 1:
                while len(data) < batch_size:
                    jobs = [self.rollouts] * (batch_size / num_process)
                    num_games += batch_size / num_process
                    pool = Pool(num_process)
                    new_data = pool.map(self.self_play, jobs)
                    pool.close()
                    pool.join()
                    data.extend(new_data)
            else:
                while len(data) < batch_size:
                    data.extend(self.self_play(self.rollouts))
                    num_games += 1

            if verbose:
                print(f'Simulated {num_games} in {str(int(time.time() - start))} seconds')
        else:
            # Get saved training data
            if verbose:
                print('Loading saved data...')
            file_names = [name for name in os.listdir(self.data_path) if os.path.isfile(name)]
            file_names.sort()
            file_names.reverse() # Reverse to get newest data first

            for name in file_names:
                path = f'{self.data_path}/{name}'

                with open(path, 'r') as reader:
                    data.extend(json.load(reader))

                if verbose:
                    print(f'File {name} loaded. Current batch_size: {len(data)}')

                if len(data) >= batch_size:
                    break
            if len(data) < batch_size:
                if verbose:
                    print('Insuficient data. Proceding to generate data with self play')
                data.extend(self.get_data(True, num_process, verbose, batch_size - len(data)))

        return data
