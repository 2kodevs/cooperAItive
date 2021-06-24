from ..player import BasePlayer

class Merge(BasePlayer):
    def __init__(self, name, players):
        super().__init__(name)
        self.players = players

    def log(self, data):
        super().log(data)
        for player in self.players:
            player.log(data)

    def reset(self, position, pieces):
        super().reset(position, pieces)
        for player in self.players:
            player.reset(position, pieces)

    def filter(self, valids=None):
        valids = super().filter(valids)
        for player in self.players:
            player.heads = self.heads
            valids = player.filter(valids)
        return valids
        

def MergeFactory(players):
    def func(name):
        composed_name = '::'.join(type(p).__name__ for p in players)
        return Merge(f'{composed_name}::{name}', [p(name) for p in players])
    return func
