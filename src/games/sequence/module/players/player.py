from ..defaults import BOARD
from ..sequence import Sequence
from ..utils import BoardViewer
import random

class BasePlayer:
    def __init__(self, name):
        self.name = name            # player name
        self.board = None           # Game board (read only) NOTE: Type is BoardViewer
        self.cards = None           # Player card (read only) NOTE: self.cards() returns an iterator
        self.color = None           # Player color
        self.history = None         # Game history
        self.position = None        # Player number
        self.can_discard = None    # Indicates if the player can change a dead card
        self.number_of_cards= None  # The number of cards per player

    def log(self, data):
        self.history.append(data)

    def step(self):
        choice = self.choice()
        if choice is not None:
            card, position = choice
            assert card in list(self.cards())
            self.can_discard = (position is not None)
            return card, position
        else:
            self.can_discard = True
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
            return Sequence.valid_moves(self.board, self.cards(), self.can_discard)
        return valids

    def reset(self, position, board, card_view, color, number_of_cards):
        self.position = position
        self.cards = card_view
        self.color = color
        self.number_of_cards = number_of_cards
        self.history = []
        self.can_discard = True
        self.board = board

    @property
    def me(self):
        return self.position

    @property
    def team(self):
        return self.color

    @staticmethod
    def from_sequence(sequence: Sequence):
        player = BasePlayer('SequencePlayer')
        player.board = BoardViewer(sequence.board)
        player.cards = sequence.cards
        player.position = sequence.current_player
        player.color = sequence.color
        player.history = sequence.logs[:]
        player.can_discard = sequence.can_discard
        player.number_of_cards = sequence.cards_per_player
        return player
        