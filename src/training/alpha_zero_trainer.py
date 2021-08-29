from trainer import Trainer

class AlphaZeroTrainer(Trainer):
    """
    Trainer manager for Alpha Zero model
    """
    def __init__(self, game, net, epochs, batch_size):
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
        super(AlphaZeroTrainer, self).__init__(game, net, epochs, batch_size)
        

    def self_play(self):
        # //TODO: Rulo
        # Only one game simulated here, and save game data.
        # This method will be called by policy_iteration,
        # use all the params that you like, or properties of the class.
        # You can change init to accept more args.
        # The NN is in self.net, you can pass it to your methods.
        pass

    def policy_iteration(self):
        raise NotImplementedError()