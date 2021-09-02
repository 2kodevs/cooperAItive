from typing import Dict
from .types import State, Action, Piece, Encoder, List, History
from ....domino import Domino


def gauss(num): 
    return (num * (num + 1)) // 2


def encoder_generator(
    max_number: int
):
    def piece_bit(a, b):
        cant = gauss(max_number + 1) - gauss(max_number + 1 - a)
        return 1 << (cant + b - a)

    def encoder(
        pieces: List[Piece],
        history: History,
        player_id: int,
    ) -> int :
        pieces_mask = 0
        for p in pieces:
            pieces_mask += piece_bit(*p)
        player = (pieces_mask, 1 << player_id, 0)

        # mapping the history
        history_encoded = []
        total_pieces = ((max_number + 1) * (max_number + 2)) // 2
        for e, *data in history:
            if e.name == 'MOVE':
                move, id, head = data
                history_encoded.append((piece_bit(*move), 1 << id, head))
            if e.name == 'PASS':
                history.append(((1 << (total_pieces + 1)), 1 << id, 0))

        # reducing the history
        tuple_bits = [total_pieces + 1, 4, 2]
        encode_number = 0
        size = 0
        for data in history_encoded:
            for bits, num in zip(tuple_bits, data):
                encode_number += (num << size)
                size += bits

        return encode_number

    return encoder

    
def state_to_list(
    state: State,
    size: int,
):
    binary_rep = bin(state)[2:]
    binary_rep = '0' * max(0, size - len(binary_rep)) + binary_rep
    return [int(x) for x in binary_rep[-1 : -(size + 1) : -1]]

