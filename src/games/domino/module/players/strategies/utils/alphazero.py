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
    Cput: int = 1,
): 
    def maker(
        domino: Domino,
        encoder: Encoder,
        player_id: int,
    ):
        s_comma_a = []
        v = None
        end_value = [0, 0, 0]
        end_value[player_id] = 1
        end_value[1 - player_id] = -1

        while v is None:
            current_player = domino.current_player
            pieces = domino.players[current_player].remaining
            history = domino.logs

            state = encoder(pieces, history, current_player)
            valids, mask = get_valids_data(domino)
            try:
                N, P, Q = data[state][:, 0], data[state][:, 1], data[state][:, 2]
                all_N = sqrt(N.sum())
                U = Cput * P * all_N / (1 + N)
                values = Q + U

                args_max = np.argwhere(values == np.max(values)).flatten()
                best_index = np.random.choice(args_max)

                s_comma_a.append((state, best_index))

                if domino.step(valids[best_index]):
                    v = end_value[domino.winner]
            except KeyError:
                [P], [v], _ = NN.predict([state], [mask])
                v = v.cpu().detach().numpy()
                size = len(P)
                npq = np.zeros((size, 3), dtype=object)
                npq[:, 1] = P.cpu().detach().numpy()
                data[state] = npq

        for state, index in s_comma_a:
            n, q = data[state][index, 0], data[state][index, 2]
            data[state][index, 0] += 1
            data[state][index, 2] = (n*q + v) / (n + 1)

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


__all__ = [
    "encoder_generator", 
    "state_to_list", 
    "get_valids_data",
]
