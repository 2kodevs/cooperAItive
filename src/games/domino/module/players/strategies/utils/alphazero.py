from .game import calc_colab
from .types import State, Action, Piece, Encoder, List, History, Any, Dict
from ....domino import Domino
from math import sqrt

import numpy as np


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
        player_id: int,
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
                id, move, head = data
                history_encoded.append((piece_bit(*move, max_number), 1 << id, head))
            if e.name == 'PASS':
                id = data[0]
                history_encoded.append((1 << total_pieces, 1 << id, 0))

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
    Coop: int,
    Cput: int,
): 
    def maker(
        domino: Domino,
        encoder: Encoder,
    ):
        s_comma_a = []
        value, c = None, None
    
        while True:
            current_player = domino.current_player
            pieces = domino.players[current_player].remaining
            history = domino.logs

            state = encoder(pieces, history, current_player)
            valids, mask = get_valids_data(domino)
            try:
                N, P, Q, C = data[state][:, 0], data[state][:, 1], data[state][:, 2], data[state][:, 3]
                all_N = sqrt(N.sum())
                U = Cput * P * all_N / (1 + N)
                values = Q + U + C 

                args_max = np.argwhere(values == np.max(values)).flatten()
                best_index = np.random.choice(args_max)

                s_comma_a.append((state, best_index, domino.current_player))

                if domino.step(valids[best_index]):
                    winner = domino.winner
                    value = lambda x: 0 if winner == -1 else [-1, 1][winner == (x & 1)]
                    c = [Coop * calc_colab(domino, player) for player in range(4)]
                    break
            except KeyError:
                [P], [v], [c] = NN.predict([state], [mask])
                v = v.cpu().detach().numpy()
                value = lambda x: v if (x & 1) == (domino.current_player & 1) else -v
                c = c.cpu().detach().numpy()
                size = len(P)
                npq = np.zeros((size, 4), dtype=object)
                npq[:, 1] = P.cpu().detach().numpy()
                data[state] = npq
                break

        for state, index, player in s_comma_a:
            v = value(player)
            N, Q, C = data[state][index, 0], data[state][index, 2], data[state][index, 3]
            data[state][index, 0] += 1
            data[state][index, 2] = (N*Q + v) / (N + 1)
            data[state][index, 3] = (N*C + c[player]) / (N + 1)

    return maker
    

def get_valids_data(
    domino: Domino,
):
    mask_size = gauss(domino.max_number + 1)

    valids = domino.valid_moves()
    if valids[0] == None:
        return valids, 1 << (mask_size * 2)

    valids.sort(key=lambda x: (x[1], piece_bit(*x[0], domino.max_number)))

    mask = [0, 0]
    for piece, head in valids:
        mask[head] += piece_bit(*piece, domino.max_number)

    return valids, mask[0] + (mask[1] << mask_size)
    

def selector_maker(
    data: Dict,
    valids: List[Action],
    turn: int,
    root: bool,
    tau_threshold: int,
    alpha: float = 0.4,
    epsilon: float = 0.25,
):
    def selector(state):
        # data = {state: [N, P, Q]}
        N = data[state][:, 0]

        if turn <= tau_threshold:
            move_values = N.astype(np.float64)
        else:
            move_values = np.zeros_like(N, dtype=np.float64)
            args_max = np.argwhere(N == np.max(N)).flatten()
            move_values[args_max] = 1
        total = move_values.sum()

        # If all actions are unexplored, move_values is uniform.
        if total == 0:
            move_values[:] = 1
            total = len(move_values)

        pi = move_values / total

        if root:
            noice = np.random.dirichlet(np.array(alpha*np.ones_like(pi)))
            pi = (1 - epsilon)*pi + epsilon*noice

        action_idx = np.random.choice(len(pi), p=pi)
        action = valids[action_idx] if valids != [] else None
        return action, pi
        
    return selector

#//TODO: Should be removed? (e1Ru1o)
def remaining_mask(remaining, max_number):
    data = [(piece_bit(*p, max_number), p) for p in remaining]
    data.sort()
    mask = sum(x for x, _ in data)
    ordered_rem = [x for _, x in data]
    return ordered_rem, mask


__all__ = [
    "encoder_generator", 
    "state_to_list", 
    "get_valids_data",
    "remaining_mask",
]
