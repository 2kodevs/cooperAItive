import numpy
from numpy.lib.twodim_base import tri
from ...player import BasePlayer
from ...player_view import PlayerView
from ....domino import Domino
from .types import *
from .game import game_hand_builder, game_data_collector, remaining_pieces
from .alphazero import get_valids_data, remaining_mask
from random import randint, choice
from .types import State, Action


def monte_carlo(
    player: BasePlayer, 
    encoder: Encoder,
    rollout: RolloutMaker,
    selector: Selector,
    handouts: int,
    rollouts: int,
    NN: Any,
) -> Tuple[State, Action] :
    # basic game information
    pieces, missing = game_data_collector(player.pieces, player.me, player.history)
    remaining = remaining_pieces(pieces, player.max_number)
    ordered_remaining, mask = remaining_mask(remaining, player.max_number)

    state = encoder(player.pieces, player.history, player.me)

    # prepare data for partner belief
    B = None
    if NN is not None:
        _, _, [B] = NN.predict([state], mask)
    r = 10 - len(pieces[player.partner])
    step = 0       # Decreasement factor
    current = 0    # Number of handouts passed
    adjustment = 0 # Actual Decreasement
    if r: step = (handouts + r - 1) // r # rounding up

    # simulations
    for _ in range(handouts):
        tries = (B is not None) * 5
        while tries:
            try:
                partner = numpy.random.choice(ordered_remaining, p=B, size=step-adjustment, replace=False)
                partner_new_numbers = set()
                for x in partner: partner_new_numbers.union(x)
                assert len(partner_new_numbers.intersection(missing[player.partner])) == 0
                new_pieces = [p[:] for p in pieces]
                new_pieces[player.partner].extend(partner)
                new_rem = list(set().union(remaining).difference(partner))
                fixed_hands = game_hand_builder(new_pieces, missing, new_rem, player.pieces_per_player)
                hand = lambda x, y: [PlayerView(h) for h in fixed_hands]
            except AssertionError:
                tries -= 1
        else:
            fixed_hands = game_hand_builder(pieces, missing, remaining, player.pieces_per_player)
            hand = lambda x, y: [PlayerView(h) for h in fixed_hands]
        current += 1
        if current == step:
            current = 0
            adjustment += 1
            adjustment = min(adjustment, step)
        
        for _ in range(rollouts):
            # New Domino Game
            domino = Domino()
            domino.reset(hand, player.max_number, player.pieces_per_player)

            # Update the history
            for e, *data in player.history:
                if e.name == "MOVE":
                    _, move, head = data
                    domino.step((move, head))
                if e.name == "PASS":
                    domino.step(None)

            # Run the rollout
            rollout(domino, encoder, player.team)

    # Select the player action
    return (state, *selector(state))
            

def rollout_maker(
    data: Dict,
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
            pieces = domino.players[current_player].remaining
            history = domino.logs

            state = encoder(pieces, history, current_player)
            valids, _ = get_valids_data(domino)
            try:
                # Check if state is explored
                _, _ = data[state]

                index = randint(0, len(valids) - 1)

                s_comma_a.append((state, index))

                if domino.step(valids[index]):
                    v = end_value[domino.winner]
            except KeyError:
                v = 0
                size = len(valids)
                data[state] = [[0] * size, [0] * size]

        for state, index in s_comma_a:
            N, Q = data[state]
            W = (Q[index] * N[index]) + v
            N[index] += 1
            Q[index] = W / N[index]

    return maker
    

def selector_maker(
    data: Dict,
    valids: List[Action],
):
    def selector(
        state: State,
    ):
        to_order = [((min(a, b), max(a, b)), h) for (a, b), h in valids]
        to_order.sort()

        _, Q = data[state]
        value = max(Q)
        filtered_data = [i for i, x in enumerate(Q) if x == value]
        return (to_order[choice(filtered_data)],)

    return selector


__all__ = ["monte_carlo"]
