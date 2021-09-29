from enum import Enum
from .utils import Color, ByPassColor, BoardViewer
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
        self.board = None               # Game board
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
        return self.players[self.current_player].view()

    @property
    def color(self):
        return self.colors[self.current_player].clone()

    @property
    def winner(self):
        assert self.logs[-1][0] == Event.WIN
        return self.logs[-1][2]

    def empty(self, i, j):
        return not self.board[i][j]

    def is_winner(self, playerId):
        return self.colors[playerId] == self.winner

    def reset(self, hand, number_of_players, players_colors, cards_per_player, win_strike=2):
        self.win_strike = win_strike
        self.cards_per_player = cards_per_player
        self.colors = [Color(c) for c in players_colors]
        self.players, self.deck = hand(number_of_players, self.cards_per_player)

        self.logs = []
        self.sequence_id = 0
        self.discard_pile = []
        self.current_player = 0
        self.can_discard = True
        self.board = [[Color() for _ in range(len(l))] for l in BOARD]
        self.board_size = sum(len(l) for l in self.board) - 4
        self.count = 0
        for i, j in CORNERS:
            self.board[i][j] = ByPassColor("X")
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
        if action is None:
            # All the cards should be dead
            for card in self.cards():
                if not self._is_dead_card(card):
                    return False
            return True

        card, pos = action
        ctype, number = card

        # check dead cards
        if pos is None:
            return self._is_dead_card(card)
        i, j = pos

        # check card in the board
        if BOARD[i][j] == card:
            # return if the board position is used
            return self.empty(i, j)

        # check if the card is a JACK
        if number is JACK:
            # check if the JACK is used correctly
            return self.empty(i, j) != (ctype in REMOVE)
        # not a valid move
        return False

    def _valid_moves(self):
       return Sequence.valid_moves(BoardViewer(self.board), self.cards(), self.can_discard)

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
            self.board[i][j].set_sequence(self.sequence_id)
        # increase the sequence number
        self.sequence_id += 1
        # update players score
        self.score[self.color.color] += 1 + (size > 5)
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
        if self.board[i][j]:
            self.board[i][j] = Color()
            self.count -= 1
            self.log((Event.REMOVE, card, pos))
            return self._next()

        # Normal play, or a JACK
        self.log((Event.PLAY, self.current_player, card, self.color, pos))
        self.board[i][j] = self.color.clone()
        self.count += 1

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
            last = self.color.clone()
            cur_i, cur_j = i, j
            while last & self.board[cur_i][cur_j]:
                if last == self.board[cur_i][cur_j]:
                    break
                last = self.board[cur_i][cur_j]
                data[idx].append((cur_i, cur_j))
                cur_i += inc_i
                cur_j += inc_j
                if not ((0 <= cur_i < 10) and (0 <= cur_j < 10)):
                    break
            data[idx] = data[idx][::-1]

        # check the other line half
        for inc_i, inc_j, idx in moves:
            last = self.color.clone()
            cur_i, cur_j = i - inc_i, j - inc_j
            if not ((0 <= cur_i < 10) and (0 <= cur_j < 10)):
                continue
            while last & self.board[cur_i][cur_j]:
                if last == self.board[cur_i][cur_j]:
                    break
                last = self.board[cur_i][cur_j]
                data[idx].append((cur_i, cur_j))
                cur_i -= inc_i
                cur_j -= inc_j
                if not ((0 <= cur_i < 10) and (0 <= cur_j < 10)):
                    break

        for line in data:
            size = len(line)
            seq = [0, 5, 9][(size >= 5) + (size >= 9)]
            self._sequence(seq, line)
              
        return self._next()

    @staticmethod
    def valid_moves(board, cards, can_discard):
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
                for (i, j), color in board:
                    if (not color.fixed) and (bool(board[i, j]) == (ctype in REMOVE)):
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
        self.players = players
        self.seq = Sequence()

        self.seq.reset(hand, len(players), players_colors, cards_per_player, win_strike)

        for i, player in enumerate(players):
            player.reset(i, BoardViewer(self.seq.board), self.seq.players[i].view(), Color(players_colors[i]), cards_per_player, len(players))
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
