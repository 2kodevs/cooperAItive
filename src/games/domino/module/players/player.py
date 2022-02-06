import random


class BasePlayer:
    def __init__(self, name):
        self.name = name
        self.position = None
        self.pieces = []
        self.history = []
        self.heads = None

    def start(self):
        return True

    def step(self, heads):
        should_pass = True

        if -1 in heads:
            # First move of the game
            should_pass = False
        else:
            for piece in self.pieces:
                if piece[0] in heads or piece[1] in heads:
                    should_pass = False
                    break

        if should_pass:
            # If player should pass because it doesn't have any valid piece
            return None

        self.heads = heads
        piece, head = self.choice()

        assert piece in self.pieces, f"Invalid piece: {piece}"
        self.pieces.remove(piece)

        return piece, head

    def valid(self, piece, head):
        """
            Check if `piece` can be put on head `head`
        """
        return self.heads[head] == -1 or self.heads[head] in piece

    def valid_moves(self):
        # List all valid moves in the form (piece, head).
        # This is put piece on head.
        valids = []

        for piece in self.pieces:
            for head in range(2):
                if self.valid(piece, head):
                    valids.append((piece, head))
        return valids
        
    def reset(self, position, pieces, max_number, timeout):
        self.position = position
        self.pieces = pieces
        self.pieces_per_player = len(pieces)
        self.max_number = max_number
        self.timeout = timeout

        self.history.clear()

    def log(self, data):
        self.history.append(data)

    def choice(self, valids=None):
        """
            Select a random move from the player filtered ones

            Return:
                piece:  (tuple<int>) Piece player is going to play. It must have it.
                head:   (int in {0, 1}) What head is it going to put the piece. This will be ignored in the first move.
        """
        valids = self.filter(valids)
        assert len(valids), "Player strategy return 0 options to select"
        return random.choice(valids)

    def score(self):
        """
            Score of current player relative to the weights of its pieces
        """
        result = 0
        for piece in self.pieces:
            result += piece[0] + piece[1]
        return result

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
            return self.valid_moves()
        return valids

    @property
    def me(self):
        return self.position

    @property
    def partner(self, position=None):
        if position is None:
            position = self.me
        return position ^ 2

    @property
    def team(self, position=None):
        """ Players 0 and 2 belong to team 0
            Players 1 and 3 belong to team 1
        """
        if position is None:
            position = self.me
        return position & 1

    @property
    def next(self, position=None):
        """ Next player to play
        """
        if position is None:
            position = self.me
        return (position + 1) & 3

    @staticmethod
    def from_domino(domino):
        player = BasePlayer('DominoPlayer')
        player.position = domino.current_player
        player.pieces = domino.players[player.me].remaining.copy()
        player.history = domino.logs[:]
        player.heads = domino.heads[:]
        player.pieces_per_player = domino.pieces_per_player
        player.max_number = domino.max_number
        return player
        