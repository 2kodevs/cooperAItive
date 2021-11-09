from ..defaults import BOARD
from ..sequence import Sequence
from ..utils import BoardViewer
import random

class BasePlayer:
    def __init__(self, name):
        self.name = name               # player name
        self.board = None              # Game board (read only) NOTE: Type is BoardViewer
        self._cards = None              # Player card (read only) NOTE: self._cards() returns an iterator
        self.history = None            # Game history
        self.position = None           # Player number
        self.win_strike = None         # Number of sequences needed to win
        self.can_discard = None        # Indicates if the player can change a dead card
        self.players_colors = None     # Colors of all the players
        self.number_of_cards = None    # The number of cards per player
        self.number_of_players = None  # The number of players in the game

    def log(self, data):
        self.history.append(data)

    def step(self):
        choice = self.choice()
        if choice is not None:
            card, position = choice
            assert card in list(self._cards())
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

            Player can access the full match history through `self.history`

            Return:
                List of:
                    cards:    (Tuple<Card, int>) Card player is going to play. It must have it.
                    Position: (Tuple<int, int>)  Selected board position to play the card
        """
        if valids is None:
            return self.valid_moves()
        return valids

    def reset(self, position, board, card_view, players_colors, number_of_cards, number_of_players, win):
        self.position = position
        self._cards = card_view
        self.players_colors = players_colors
        self.number_of_cards = number_of_cards
        self.number_of_players = number_of_players
        self.history = []
        self.can_discard = True
        self.board = board
        self.win_strike = win

    def valid_moves(self):
        return Sequence.valid_moves(self.board, self.cards, self.can_discard, self.color) 

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
        player.board = BoardViewer(sequence.board)
        player.position = sequence.current_player
        player._cards = sequence.players[player.me].view()
        player.players_colors = sequence.colors
        player.history = sequence.logs[:]
        player.can_discard = sequence.can_discard
        player.number_of_cards = sequence.cards_per_player
        return player
        