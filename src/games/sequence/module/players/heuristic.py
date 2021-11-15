from ..utils import Piece
from ..defaults import BOARD
from .player import BasePlayer
from .utils.game import move_score


class Heuristic(BasePlayer):
    def __init__(self, name):
        super().__init__(f"Heuristc::{name}")

    def filter(self, valids=None):
        if valids is None:
            valids = self.valid_moves()

        if valids[0] is None:
            return valids

        board = [[Piece() for _ in range(len(l))] for l in BOARD]
        for (i, j), piece in self.board:
            board[i][j] = piece

        best = -float("inf")
        options, discards = [], []
        for move in valids:
            _, pos = move
            if pos is None:
                discards.append(move)
                continue
            colab = move_score(board, self.color, *pos)
            if colab > best:
                best = colab
                options = []
            if colab == best:
                options.append(move)
                    
        return super().filter(valids=[*options, *discards])
