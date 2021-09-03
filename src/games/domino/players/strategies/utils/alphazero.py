from typing import Dict
from .types import State, Action, Piece, Encoder, List, History, Any
from ....domino import Domino
from math import sqrt
from random import choice


def gauss(num): 
    return (num * (num + 1)) // 2

def piece_bit(a, b, max_number):
    if a > b:
        a, b = b, a
    cant = gauss(max_number + 1) - gauss(max_number + 1 - a)
    return 1 << (cant + b - a)

def encoder_generator(
    max_number: int
):
    def encoder(
        pieces: List[Piece],
        history: History,
        player_id: int, # //TODO: check that playerId is in range(4)
    ) -> int :
        pieces_mask = 0
        for p in pieces:
            pieces_mask += piece_bit(*p, max_number)
        player = (pieces_mask, 1 << player_id, 0)

        # mapping the history
        history_encoded = [player]
        total_pieces = ((max_number + 1) * (max_number + 2)) // 2
        for e, *data in history:
            if e.name == 'MOVE':
                move, id, head = data
                history_encoded.append((piece_bit(*move, max_number), 1 << id, head))
            if e.name == 'PASS':
                history_encoded.append(((1 << (total_pieces + 1)), 1 << id, 0))

        # reducing the history
        tuple_bits = [total_pieces + 1, 4, 2]
        encoded_number = 0
        size = 0
        for data in history_encoded:
            for bits, num in zip(tuple_bits, data):
                encoded_number += (num << size)
                size += bits

        return encoded_number

    return encoder

    
def state_to_list(
    state: State,
    size: int,
):
    binary_rep = bin(state)[2:]
    binary_rep = '0' * max(0, size - len(binary_rep)) + binary_rep
    return [int(x) for x in binary_rep[-1 : -(size + 1) : -1]]


def rollout_maker(
    data: Dict,
    NN: Any,
    Cput: int = 1,
): 
    def maker(
        domino: Domino,
        encoder: Encoder,
        playerId: int,
    ):
        s_comma_a = []
        v = None
        end_value = [0, 0, 0]
        end_value[playerId] = 1
        end_value[1 - playerId] = -1

        while v is None:
            current_player = domino.current_player
            pieces = domino.players[current_player].pieces
            history = domino.logs

            state = encoder(pieces, history, current_player)
            try:
                N, P, Q = data[state]
                all_N = sqrt(sum(N))

                values = [
                    (Cput * p * all_N + w) / (1 + n) # utility value
                    for n, p, w in zip(N, P, Q)
                ]

                best = max(values)
                filtered_values = [i for i, x in enumerate(values) if x == best]
                index = choice(filtered_values)

                s_comma_a.append((state, index))

                valids = domino.valid_moves()
                if valids[0] != None:
                    valids.sort(key=lambda x: (x[1], piece_bit(*x[0], domino.max_number)))
                done = domino.step(valids[index])
                if done:
                    v = end_value[domino.winner]
            except KeyError:
                P, v = NN(state)
                size = len(P)
                data[state] = [[0] * size, P, [0] * size]

        for (N, _, Q), index in s_comma_a:
            W = (Q[index] * N[index]) + v
            N[index] += 1
            Q[index] = W / N[index]

    return maker
    