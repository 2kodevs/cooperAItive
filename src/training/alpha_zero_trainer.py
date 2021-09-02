from ..models import alpha_zero_net as Net
from .trainer import Trainer

import numpy as np
import random
class AlphaZeroTrainer(Trainer):
    """
    Trainer manager for Alpha Zero model
    """
    def __init__(self, game: Domino, net: Net, batch_size):
        """
        param game:
            Manager of the game in which the agent is a player
        param net: nn.Module
            Neural Network to train
        param epochs: int
            Number of training iterations
        param batch_size: int
            Size of training data used for epoch
        """
        self.game = game
        self.net = net
        self.batch_size = batch_size
        

    def self_play(self, rollouts):
        # //TODO: Rulo
        # Only one game simulated here, and save game data.
        # This method will be called by policy_iteration,
        # use all the params that you like, or properties of the class.
        # You can change init to accept more args.
        # The NN is in self.net, you can pass it to your methods.
        pass

    def policy_iteration(self, epoch, rollouts, verbose=False):
        data = []
        while len(data) < self.batch_size:
            data.extend(self.self_play(rollouts))
            
        batch = random.sample(data, self.batch_size)
        loss = self.net.train(batch)
        Trainer.adjust_learning_rate(epoch, self.net.optimizer)
        return loss
