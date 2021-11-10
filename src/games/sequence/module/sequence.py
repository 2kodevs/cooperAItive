from enum import Enum
from .utils import Piece, ByPassPiece, BoardViewer, lines_collector
from .defaults import *
from random import shuffle


class Event(Enum):
    # Report beginning
    # params: ()
    NEW_GAME = 0

    # Player use a put a new chip on the board
    # params: (playerId, card, color, position)
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

    # Report deck refill
    REFILL_DECK = 7


class Sequence:
    """
    Instance that contains the logic of a single match.
    """
    def __init__(self):
        self.deck = None                # Game deck
        self.logs = None                # Game history
        self._board = None              # Game board
        self.count = None               # Number of positions used of the board
        self.score = None               # Score per color (i.e Number of consecutive sequences)
        self.colors = None              # Players color
        self.players = None             # Players list
        self.board_size = None          # Amount of positions of the board
        self.win_strike = None          # Number of consecutive sequences needed to win the game
        self.sequence_id = None         # Incremental id for identifiying each sequence
        self.can_discard = None         # Indicate if the current player can discard a card
        self.discard_pile = None        # Players discarted cards
        self.current_player = None      # Id of the current player
        self.cards_per_player = None    # Number of cards per player

    def log(self, data):
        self.logs.append(data)

    @property
    def cards(self):
        return self.players[self.current_player].view()()

    @property
    def color(self):
        return self.colors[self.current_player]

    @property
    def partners(self):
        return self._partner(self.current_player)

    @property
    def board(self):
        return BoardViewer(self._board)

    @property
    def winner(self):
        assert self.logs[-1][0] == Event.WIN
        return self.logs[-1][2]

    def _partners(self, player):
        color = self.colors[player] 
        for i, c in enumerate(self.colors):
            if i != player and c == color:
                yield i

    def empty(self, i, j):
        return not self._board[i][j]

    def is_winner(self, playerId):
        return self.colors[playerId] == self.winner

    def reset(self, hand, number_of_players, players_colors, cards_per_player, win_strike=2):
        self.win_strike = win_strike
        self.colors = players_colors
        self.cards_per_player = cards_per_player
        self.players, self.deck = hand(number_of_players, cards_per_player)

        self.logs = []
        self.sequence_id = 0
        self.discard_pile = []
        self.current_player = 0
        self.can_discard = True
        self._board = [[Piece() for _ in range(len(l))] for l in BOARD]
        self.board_size = sum(len(l) for l in self._board) - 4
        self.count = 0
        for i, j in CORNERS:
            self._board[i][j] = ByPassPiece("X")
        self.score = {i:0 for i in set(players_colors)}

        self.log((Event.NEW_GAME,))

    def _is_dead_card(self, card):
        _, number = card
        if number is JACK:
            return False # //TODO: Not sure about it
        for (i, j) in CARDS_POSITIONS[card]:
            if self.empty(i, j):
                return False
        return True

    def check_valid(self, action):
        return action in self._valid_moves()

    def _valid_moves(self):
       return Sequence.valid_moves(self.board, self.cards, self.can_discard, self.color)

    def _is_over(self):
        if max(self.score.values()) >= self.win_strike:
            self.log((Event.WIN, self.current_player, self.color))
            return True
        if self.count == self.board_size:
            for e, *data in self.logs:
                if e is Event.SEQUENCE:
                    player, color, _ = data
                    self.log((Event.WIN, player, color))
                    break
            else:
                self.log((Event.WIN, None, None))
            return True
        return False

    def _sequence(self, size, data):
        # skip empty findings
        if not size:
            return
        # set the sequence number in each board position
        for i, j in data[:size]:
            self._board[i][j].set_sequence(self.sequence_id)
        # increase the sequence number
        self.sequence_id += 1
        # update players score
        self.score[self.color] += 1 + (size > 5)
        # Report the sequence
        self.log((Event.SEQUENCE, self.current_player, self.color, size))

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
            self.log((Event.REFILL_DECK,))
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
        if not self.check_valid(action):
            raise ValueError(f"Invalid move ({action})")

        # Check PASS
        if action is None:
            self.log((Event.PASS, self.current_player))
            return self._next()

        card, pos = action
        self._discard(card)
        # Check DISCARD
        if pos is None:
            self.log((Event.DISCARD, self.current_player, card))
            self.can_discard = False # Discard only one card per turn
            return False # Game not finished, still the current_player turn
        
        i, j = pos

        # check REMOVE action
        if self._board[i][j]:
            self._board[i][j] = Piece()
            self.count -= 1
            self.log((Event.REMOVE, self.current_player, card, pos))
            return self._next()

        # Normal play, or a JACK
        self.log((Event.PLAY, self.current_player, card, self.color, pos))
        self._board[i][j] = Piece(self.color)
        self.count += 1

        # check for sequences

        data = lines_collector(self._board, self.color, i, j)

        for line in data:
            size = len(line)
            seq = [0, 5, 9][(size >= 5) + (size >= 9)]
            self._sequence(seq, line)
              
        return self._next()

    @staticmethod
    def valid_moves(board, cards, can_discard, pcolor):
        # List all valid moves in the form (card, position).
        valids = []

        for card in cards:
            try:
                is_dead = True
                for i, j in CARDS_POSITIONS[card]:
                    if not board[i, j]:
                        is_dead = False
                        valids.append((card, (i, j)))
                if can_discard and is_dead:
                    valids.append((card, None))
            except KeyError:
                ctype, number = card
                assert number is JACK, f"Unexpected card number ({number})"
                if ctype in REMOVE:
                    for (i, j), piece in board:
                        if piece and piece.color != pcolor and not piece.fixed:
                            valids.append((card, (i, j)))
                else:
                    for (i, j), piece in board:
                        if not (piece.bypass() or piece):
                            valids.append((card, (i, j)))
        return valids if valids else [None]


class SequenceManager:
    @property
    def cur_player(self):
        return self.players[self.seq.current_player]

    def feed_logs(self):
        while self.logs_transmitted < len(self.seq.logs):
            data = self.seq.logs[self.logs_transmitted]
            for player in self.players:
                player.log(data)
            self.logs_transmitted += 1

    def init(self, hand, players, players_colors, cards_per_player, win_strike=2):
        self.logs_transmitted = 0
        self.players = [player(i) for i, player in zip("0123", players)]
        self.seq = Sequence()

        self.seq.reset(hand, len(players), players_colors, cards_per_player, win_strike)

        for i, player in enumerate(self.players):
            player.reset(
                i, 
                self.seq.board, 
                self.seq.players[i].view(), 
                players_colors[:], 
                cards_per_player, 
                len(players),
                win_strike,
            )
        self.feed_logs()

    def step(self, fixed_action=False, action=None):
        if not fixed_action:
            action = self.cur_player.step()
        done = self.seq.step(action)
        self.feed_logs()
        return done

    def run(self, hand, players, players_colors, cards_per_player, win_strike=2):
        self.init(hand, players, players_colors, cards_per_player, win_strike)

        while not self.step(): pass

        return self.seq.winner
