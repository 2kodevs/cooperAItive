class Trainer:
    """
    Abstract class of trainer instances for models
    """

    def self_play(self):
        raise NotImplementedError()

    def policy_iteration(self):
        raise NotImplementedError()
