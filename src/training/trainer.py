class Trainer:
    """
    Abstract class of trainer instances for models
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
        self.game = game
        self.net = net
        self.epochs = epochs
        self.batch_size = batch_size

    def self_play(self):
        raise NotImplementedError()

    def policy_iteration(self):
        raise NotImplementedError()

    #Create a learning rate adjustment function that divides the learning rate by 10 every 30 epochs
    @staticmethod
    def adjust_learning_rate(epoch, optimizer):

        lr = 0.001

        if epoch > 180:
            lr = lr / 1000000
        elif epoch > 150:
            lr = lr / 100000
        elif epoch > 120:
            lr = lr / 10000
        elif epoch > 90:
            lr = lr / 1000
        elif epoch > 60:
            lr = lr / 100
        elif epoch > 30:
            lr = lr / 10

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr