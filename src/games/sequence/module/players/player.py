from ..defaults import BOARD
from ..sequence import Sequence, SequenceView
from ..utils import BoardViewer
import random

class BasePlayer:
    def __init__(self, name):
        self.name = name        # player name
        self.seq = None         # The player game view
        self._cards = None      # Player card (read only) NOTE: self._cards() returns an iterator
        self.history = None     # Game history
        self.position = None    # Player number

    def log(self, data):
        self.history.append(data)

    def step(self):
        choice = self.choice()
        if choice is not None:
            card, position = choice
            assert card in list(self._cards())
            return card, position
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

            Player can access the full match history through `self.history`

            Return:
                List of:
                    cards:    (Tuple<Card, int>) Card player is going to play. It must have it.
                    Position: (Tuple<int, int>)  Selected board position to play the card
        """
        if valids is None:
            return self.valid_moves()
        return valids

    def reset(self, position, cards_view, sequence: SequenceView):
        self.position = position
        self._cards = cards_view
        self.seq = sequence
        self.history = []

    def valid_moves(self):
        return Sequence.valid_moves(self.board, self.cards, self.can_discard, self.color) 

    @property
    def players_colors(self):
        return self.seq.colors

    @property
    def colors(self):
        return self.players_colors

    @property
    def is_current(self):
        return self.me == self.seq.player

    @property
    def win_strike(self):
        return self.seq.strike

    @property
    def board(self):
        return self.seq.board

    @property
    def can_discard(self):
        return self.seq.discard

    @property
    def number_of_cards(self):
        return self.seq.cards

    @property
    def number_of_players(self):
        return self.seq.players

    @property
    def pile(self):
        return self.seq.pile

    @property
    def score(self):
        return self.seq.score

    @property
    def me(self):
        return self.position

    @property
    def team(self):
        return self.color

    @property
    def color(self):
        return self.players_colors[self.position]

    @property
    def cards(self):
        return self._cards()

    @staticmethod
    def from_sequence(sequence: Sequence):
        player = BasePlayer('SequencePlayer')
        player.position = sequence.current_player
        player._cards = sequence.players[player.me].view()
        player.history = sequence.logs[:]
        player.seq = sequence.view
        return player
        