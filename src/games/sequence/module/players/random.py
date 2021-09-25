from .player import BasePlayer

class Random(BasePlayer):
    def __init__(self, name):
        super().__init__(f'Random::{name}')
