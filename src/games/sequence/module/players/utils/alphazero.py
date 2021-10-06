from typing import Tuple
from ...utils import BoardViewer
from ...defaults import ALL_CARDS_MAPPING, CORNERS, JACK
from ...sequence import Sequence
from ..player import BasePlayer
from .types import Action, GameData, Position, RolloutMaker, Selector, State, Encoder, Any, Dict, List, Card
from .game import get_discard_pile
from math import sqrt
import numpy as np


def table_bit(i: int, j: int) -> int:
    return i * 10 + j


def encode_board(board: BoardViewer, color: Any = -1) -> int:
    masks = [0]
    colors = {color: 0}
    for (i, j), item in board:
        if item:
            if not item.color in colors:
                colors[item.color] = len(colors)
                masks.append(0)
            masks[colors[item.color]] |= 1 << table_bit(i, j)           
    return masks


def encode_cards(cards: List[Card]) -> int:
    mask = 0
    data = {c:0 for c in cards}
    for c in cards:
        mask |= 1 << table_bit(*ALL_CARDS_MAPPING[c][data[c]])
        data[c] += 1
    return mask


def adjust_shifting(pos: Position) -> int:
    return len([1 for x in CORNERS if x < pos])


def encode_valids(valids: List[Action]) -> int:
    if valids[0] is None:
        return 1 << 192
    mask = 0
    for (_, num), pos in valids:
        cur_bit = 1 << (table_bit(*pos) - adjust_shifting(pos))
        if num is JACK: mask |= cur_bit << 96
        else : mask |= cur_bit
    return mask


def encode(
    player: GameData,
    discard_pile: List[Card],
) -> State :       
    boards = encode_board(player.board, player.color)
    cards = encode_cards(list(player.cards))
    pile = encode_cards(discard_pile)
    offset = 0
    state = 0
    for mask in [*boards, cards, pile]:
        state += (mask << offset)
        offset += 104
    return state

    
def state_to_list(
    state: State,
    size: int,
) -> List[int]:
    binary_rep = bin(state)[2:]
    binary_rep = '0' * max(0, size - len(binary_rep)) + binary_rep
    return [int(x) for x in binary_rep[-1 : -(size + 1) : -1]]


def rollout_maker(
    data: Dict,
    NN: Any,
    Cput: int = 1,
) -> RolloutMaker: 
    def maker(
        sequence: Sequence,
        encoder: Encoder,
    ) -> None:
        s_comma_a = []
        v = None
        end_value = {c.color:(sequence.color == c.color) for c in sequence.colors}

        while v is None:
            state = encoder(sequence, get_discard_pile(sequence.logs))
            valids = sequence._valid_moves()
            mask = encode_valids(valids)
            try:
                N, P, Q = data[state][:, 0], data[state][:, 1], data[state][:, 2]
                all_N = sqrt(N.sum())
                U = Cput * P * all_N / (1 + N)
                values = Q + U

                args_max = np.argwhere(values == np.max(values)).flatten()
                best_index = np.random.choice(args_max)

                s_comma_a.append((state, best_index))

                if sequence.step(valids[best_index]):
                    v = end_value[sequence.winner]
            except KeyError:
                [P], [v] = NN.predict([state], [mask])
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


def selector_maker(
    data: Dict,
    valids: List[Action],
    turn: int,
    root: bool,
    tau_threshold: int,
    alpha: float = 0.4,
    epsilon: float = 0.25,
) -> Selector:
    def selector(state: State) -> Tuple[Action, List[Any]]:
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
    "encode", 
    "state_to_list", 
]
