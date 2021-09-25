from ..defaults import BOARD
from ..utils import Color
from ..sequence import Sequence
import random

class BasePlayer:
    def __init__(self, name):
        self.name = name
        self.board = None
        self.cards = None
        self.color = None
        self.history = None
        self.position = None
        self.cand_discard = None
        self.number_of_cards= None

    def log(self, data):
        self.history.append(data)

    def step(self):
        choice = self.choice()
        if choice is not None:
            card, position = choice
            assert card in list(self.cards())
            self.cand_discard = (position is not None)
        else:
            self.cand_discard = True
            return None

    def choice(self, valids=None):
        """
            Select a random move from the player filtered ones
        """
        valids = self.filter(valids)
        assert len(valids), "Player strategy return 0 options to select"
        return random.choice(valids)

    def filter(self, valids=None):
        """
            Logic of each agent. This function given a set of valids move select the posible options. 
            Notice that rules force player to always make a move whenever is possible.

            Player can access to current heads using `self.heads` or even full match history
            through `self.history`

            Return:
                List of:
                    piece:  (tuple<int>) Piece player is going to play. It must have it.
                    head:   (int in {0, 1}) What head is it going to put the piece. This will be ignored in the first move.
        """
        if valids is None:
            return Sequence.valid_moves(self.board, list(self.cards()), self.cand_discard)
        return valids

    def reset(self, position, card_view, color, number_of_cards):
        self.position = position
        self.cards = card_view
        self.color = color
        self.number_of_cards = number_of_cards
        self.history = []
        self.cand_discard = True
        self.board = [[Color() for _ in range(len(l))] for l in BOARD]

    @property
    def me(self):
        return self.position

    @property
    def team(self):
        return self.color

    @staticmethod
    def from_sequence(sequence):
        player = BasePlayer('SequencePlayer')
        player.position = sequence.current_player
        player.cards = sequence.players[player.me].view()
        player.history = sequence.logs[:]
        player.pieces_per_player = sequence.pieces_per_player
        player.color = sequence.colors[player.me]
        return player
        