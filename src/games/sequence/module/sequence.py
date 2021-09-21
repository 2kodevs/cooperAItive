from enum import Enum
from .utils import Color, ByPassColor 
from .defaults import *
from random import shuffle


class Event(Enum):
    # Report beginning
    # params: ()
    NEW_GAME = 0

    # Player use a put a new chip on the board
    # params: (playerId, card, position)
    PLAY = 1

    # Player remove a chip from the board
    # params: (playerId, card, position)
    REMOVE = 2

    # Player complete a sequence
    # params: (playerId, color, size)
    SEQUENCE = 3

    # Player change a card
    # params: (playerId, card)
    DISCARD = 4

    # Player pass
    # params: (playerId)
    PASS = 5

    # Report winner
    # params: (playerId, color)
    WIN = 6


class Sequence:
    """
    Instance that contains the logic of a single match.
    """
    def __init__(self):
        self.deck = None
        self.logs = None
        self.board = None
        self.score = None
        self.colors = None
        self.sealed = None
        self.players = None
        self.win_strike = None
        self.sequence_id = None
        self.can_discard = None
        self.discard_pile = None
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

    def reset(self, hand, number_of_players, players_colors, pieces_per_player, win_strike=2):
        self.win_strike = win_strike
        self.pieces_per_player = pieces_per_player
        self.colors = [Color(c) for c in players_colors]
        self.players, self.deck = hand(number_of_players, self.pieces_per_player)

        self.logs = []
        self.sequence_id = 0
        self.discard_pile = []
        self.current_player = 0
        self.can_discard = True
        self.board = [[Color() for _ in range(len(l))] for l in BOARD]
        for i, j in CORNERS:
            self.board[i][j] = ByPassColor(-1)
        self.score = {i:0 for i in set(players_colors)}

        self.log(Event.NEW_GAME)

    def check_valid(self, action):
        card, pos = action
        ctype, number = card

        # check dead cards
        if pos is None:
            if number is JACK:
                return False
            for i, j in CARDS_POSITIONS[card]:
                if not self.board[i][j]:
                    return False
            return True
        i, j = pos

        # check card in the board
        if BOARD[i][j] == card:
            # return if the board position is used
            return self.empty(i, j)

        # check if the card is a JACK
        if number == JACK:
            # check if the JACK is used correctly
            return self.empty(i, j) == (ctype in REMOVE)
        # not a valid move
        return False

    def valid_moves(self):
        # List all valid moves in the form (card, position).
        valids = []

        cards = self.players[self.current_player].cards
        for card in cards:
            try:
                for (i, j) in CARDS_POSITIONS[card]:
                    if not self.board[i][j]:
                        valids.append((card, (i, j)))
                else:
                    if self.can_discard:
                        valids.append((card, None))
            except KeyError:
                ctype, number = card
                assert number == JACK, "Unexpected card number"
                for i, row in enumerate(self.board):
                    for j, color in enumerate(row):
                        if (not color.fixed) and (bool(color) == (ctype in REMOVE)):
                            valids.append((card, (i, j)))
        return valids

    def _is_over(self):
        if max(self.score) >= self.win_strike:
            player = self.current_player
            self.log(Event.WIN, player, self.colors[player].color)
            return True
        return False

    def _sequence(self, size, data):
        # skip empty findings
        if not size:
            return
        # set the sequence number in each board position
        for i, j in data[:size]:
            self.board[i][j].set_sequence(self.sequence_id)
        # increase the sequence number
        self.sequence_id += 1
        # update players score
        player = self.current_player
        color = self.colors[player]
        new_score = {x:0 for x in self.score}
        new_score[color.color] = self.score[color.color] + 1 + (size > 5)
        # Report the sequence
        self.log((Event.SEQUENCE, player, color.color, size))

    def _next(self):
        over = self._is_over()
        self.can_discard = True
        self.current_player = (self.current_player + 1) % len(self.players)
        return over

    def _discard(self, card):
        player = self.players[self.current_player]
        player.remove(card)
        self.discard_pile.append(card)
        if not self.deck:
            self.deck = self.discard_pile[:]
            self.discard_pile = []
            shuffle(self.deck)
        if self.deck:
            player.draw(self.deck.pop())

    def step(self, action):
        """
        `action` must be a tuple of the form `(card, position)` where: 
        
        * card is `(enum, int)` 

        * position is None if the player is discating the card.

        * position is (int, int) if the player is playing the card

        raise ValueError if it's an invalid move.
        """
        # Check PASS
        if action is None:
            self.log(Event.PASS, self.current_player)
            return self._next(False)

        if not self.check_valid(action):
            raise ValueError("Invalid move.")

        card, pos = action
        # Check DISCARD
        if pos is None:
            self.log(Event.DISCARD, self.current_player, card)
            self._discard(card)
            self.can_discard = False # Discard only one card per turn
            return False # Game not finished, still the current_player turn
        
        i, j = pos
        color = self.colors[self.current_player]

        # check REMOVE action
        if self.board[i][j]:
            self.board[i][j] = Color()
            self.log(Event.REMOVE, card, pos)
            return self._next()

        # Normal play, or a JACK
        self.log(Event.PLAY, card, pos)
        self._discard(card)
        self.board[i][j] = color.clone()

        # check for sequences
        data = [[], [], [], []] # one per direction
        moves = [
            # (i, j, data)
            (-1, -1, 0),
            (0, -1, 1),
            (1, -1, 2),
            (1, 0, 3),
        ]
        # check a half of the line
        for inc_i, inc_j, idx in moves:
            last = color.clone()
            cur_i, cur_j = i, j
            while last & self.board[cur_i][cur_j]:
                if last == self.board[cur_i][cur_j]:
                    break
                data[idx].append((cur_i, cur_j))
                cur_i += inc_i
                cur_j += inc_j
                if not ((0 <= cur_i < 10) and (0 <= cur_j < 10)):
                    break
            data[idx] = data[idx][::-1]

        # check the other line half
        for inc_i, inc_j, idx in moves:
            last = color.clone()
            cur_i, cur_j = i - inc_i, j - inc_j
            if not ((0 <= cur_i < 10) and (0 <= cur_j < 10)):
                continue
            while last & self.board[cur_i][cur_j]:
                if last == self.board[cur_i][cur_j]:
                    break
                data[idx].append((cur_i, cur_j))
                cur_i -= inc_i
                cur_j -= inc_j
                if not ((0 <= cur_i < 10) and (0 <= cur_j < 10)):
                    break

        for line in data:
            size = len(line)
            seq = [0, 5, 9][(size >= 5) + (size >= 9)]
            self.sequence(seq, line)
              
        return self.next()
