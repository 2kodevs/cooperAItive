from random import choice, randint
from ..player import BasePlayer
from ...sequence import Event, Sequence
from .types import *
from .game import get_discard_pile, fixed_hand


def monte_carlo(
    player: BasePlayer,
    encoder: Encoder,
    rollout: RolloutMaker,
    selector: Selector,
    handouts: int,
    rollouts: int,
) -> Tuple[State, Action]:
    # basic game information
    discard_pile = get_discard_pile(player.history)
    hand = fixed_hand(player.cards, discard_pile, player.me)
    score = {c:0 for c in player.players_colors}
    for e, *data in player.history:
        if e is Event.SEQUENCE:
            color, size = data
            score[color] += 1 + (size > 5)

    # simulations
    for _ in range(handouts):
        for _ in range(rollouts):
            # New Sequence Game
            seq = Sequence()
            seq.reset(
                hand, 
                player.number_of_players, 
                player.players_colors,
                player.number_of_cards,
                player.win_strike,
            )
            # Update the game
            for (i, j), color in player.board:
                seq.board[i][j] = color
                seq.count += bool(color)
            seq.discard_pile = discard_pile[:]
            for log in player.history:
                seq.log(log)
            seq.can_discard = player.can_discard
            seq.current_player = player.position
            seq.score = score.copy()

            # Run the rollout
            rollout(seq, encoder, player.team)

    # Select the player action
    state = encoder(player)
    return (state, *selector(state))
            
    
def rollout_maker(
    data: Dict,
): 
    def maker(
        sequence: Sequence,
        encoder: Encoder,
    ):
        s_comma_a = []
        v = None
        end_value = {c.color:(sequence.color == c.color) for c in sequence.colors}

        while v is None:
            state = encoder(sequence, get_discard_pile(sequence.logs))
            valids = sequence._valid_moves()
            try:
                # Check if state is explored
                _, _ = data[state]

                index = randint(0, len(valids) - 1)

                s_comma_a.append((state, index))

                if sequence.step(valids[index]):
                    v = end_value[sequence.winner]
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
        _, Q = data[state]
        value = max(Q)
        filtered_data = [i for i, x in enumerate(Q) if x == value]
        return (valids[choice(filtered_data)],)

    return selector


__all__ = ["monte_carlo"]
