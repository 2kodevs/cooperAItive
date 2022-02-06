from ..player import BasePlayer
import requests

class Remote(BasePlayer):
    def __init__(self, name, endpoint):
        super().__init__(f"Remote::{name}")
        self.endpoint = endpoint

    def start(self):
        return requests.post(f'{self.endpoint}/start').json()

    def step(self, heads):
        move = requests.post(f'{self.endpoint}/step', json=heads).json()
        if move is None: return None
        return tuple(move[0]), move[1]

    def reset(self, pos, pieces, max_number, timeout, score):
        requests.post(f'{self.endpoint}/reset', json={
            "position": pos,
            "pieces": pieces,
            "max_number": max_number,
            "timeout": timeout,
            "score": score,
        })

    def log(self, log):
        event, *args = log
        requests.post(f'{self.endpoint}/log', json=[event.value, *args])
