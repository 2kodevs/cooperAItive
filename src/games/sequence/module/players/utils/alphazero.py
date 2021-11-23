from ...sequence import Sequence
from .types import Action, RolloutMaker, Selector, State, Encoder, Any, Dict, List, Tuple
from .game import calc_colab, encode_valids
from math import sqrt
import numpy as np


def rollout_maker(
    data: Dict,
    NN: Any,
    Coop: int,
    Cput: int,
) -> RolloutMaker: 
    def maker(
        sequence: Sequence,
        encoder: Encoder,
    ) -> None:
        s_comma_a = []
        value, c = None, None

        while True:
            state = encoder(sequence, sequence.discard_pile)
            valids = sequence._valid_moves()
            mask = encode_valids(valids)
            try:
                N, P, Q, C = data[state][:, 0], data[state][:, 1], data[state][:, 2], data[state][:, 3]
                all_N = sqrt(N.sum())
                U = Cput * P * all_N / (1 + N)
                values = Q + U + C 

                args_max = np.argwhere(values == np.max(values)).flatten()
                best_index = np.random.choice(args_max)

                s_comma_a.append((state, best_index, sequence.current_player))

                if sequence.step(valids[best_index]):
                    value = lambda x: 0 if sequence.winner is None else [-1, 1][sequence.is_winner(x)]
                    c = [Coop * calc_colab(sequence, player) for player in range(4)]
                    break
            except KeyError:
                [P], [v], [c] = NN.predict([state], [mask])
                v = v.cpu().detach().numpy()
                player_color = sequence.colors[sequence.current_player]
                value = lambda x: v if player_color == sequence.colors[x] else -v
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


def selector_maker(
    data: Dict,
    valids: List[Action],
    turn: int,
    root: bool,
    tau_threshold: int,
    alpha: float = 1.0,
    epsilon: float = 0.50,
) -> Selector:
    def selector(state: State) -> Tuple[Action, List[Any]]:
        # data = {state: [N, P, Q, C]}
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
    "table_bit",
    "selector_maker",
    "encode_valids",
]
