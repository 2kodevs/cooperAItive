from enum import Enum


class Event(Enum):
    # Report beginning
    # params: ()
    NEW_GAME = 0

    # Player don't have any valid piece
    # params: (playerId, card, position)
    PLAY = 1

    # Player makes a move
    # params: (playerId, card, position)
    REMOVE = 2

    # Last piece of a player is put
    # params: (playerId, color)
    SEQUENCE = 3

    # Report winner
    # params: (playerId, color)
    WIN = 4


class Sequence:
    """
    Instance that contains the logic of a single match.
    """
    def __init__(self):
        self.logs = None
        self.board = None
        self.colors = None
        self.players = None
        self.current_player = None
        self.pieces_per_player = None

    def log(self, data):
        self.logs.append(data)

    def get_cards(self):
        return [player.cards for player in self.players]

    @property
    def winner(self):
        assert self.logs[-1][0] == Event.WIN
        return self.logs[-1][0]

    def is_winner(self, playerId):
        return self.colors[playerId] == self.colors[self.winner]

    def reset(self, hand, number_of_players, players_colors, pieces_per_player):
        self.pieces_per_player = pieces_per_player
        self.colors = players_colors
        self.players = hand(number_of_players, self.pieces_per_player)

        self.logs = []
        self.current_player = 0

        self.log(Event.NEW_GAME)

    def check_valid(self, action):
        pass

