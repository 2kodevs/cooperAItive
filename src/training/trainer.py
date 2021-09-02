class Trainer:
    """
    Abstract class of trainer instances for models
    """

    def self_play(self):
        raise NotImplementedError()

    def policy_iteration(self):
        raise NotImplementedError()

    #Create a learning rate adjustment function that divides the learning rate by 10 every 30 epochs
    @staticmethod
    def adjust_learning_rate(epoch, optimizer):

        lr = 0.02

        if epoch > 500:
            lr = lr / 1000
        elif epoch > 300:
            lr = lr / 100
        elif epoch > 100:
            lr = lr / 10

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr