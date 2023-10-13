from ..player import BasePlayer
import requests

from urllib.parse import urljoin

class Remote(BasePlayer):
    def __init__(self, name, server):
        super().__init__(f"Remote::{name}")
        self.server = server

    def call(self, method, endpoint, json={}):
        url = urljoin(self.server, endpoint)
        while True:
            try:
                return method(url, json=json)
            except requests.exceptions.ConnectionError:
                pass # always retry

    def start(self):
        return bool(self.call(requests.get, 'start').json()['start'])

    def step(self, heads):
        move = self.call(requests.post, 'step', heads).json()
        if move['piece'] is None: return None
        return tuple(move['piece']), move['head']

    def reset(self, pos, pieces, max_number, timeout, score):
        self.call(requests.post, 'reset', json={
            "position": pos,
            "pieces": pieces,
            "max_number": max_number,
            "timeout": timeout,
            "score": score,
        })

    def log(self, log):
        event, *args = log
        self.call(requests.post, 'log', json=[event.name, *args])
