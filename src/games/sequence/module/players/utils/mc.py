from random import choice, randint
from ..player import BasePlayer
from ...sequence import Sequence
from .types import *
from .game import fixed_hand


def monte_carlo(
    player: BasePlayer,
    encoder: Encoder,
    rollout: RolloutMaker,
    selector: Selector,
    handouts: int,
    rollouts: int,
) -> Tuple[State, Action]:
    # basic game information
    discard_pile = player.pile
    score = player.score

    # simulations
    for _ in range(handouts):
        hand = fixed_hand(list(player.cards), discard_pile, player.me, player.number_of_cards)
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
            for (i, j), piece in player.board:
                seq._board[i][j] = piece
                seq.count += bool(piece)
            seq.logs = player.history[:]
            seq.discard_pile = discard_pile[:]
            seq.can_discard = player.can_discard
            seq.current_player = player.position
            seq.score = score.copy()
            seq.sequence_id = sum(score.values())

            rollout(seq, encoder)

    # Select the player action
    state = encoder(player, discard_pile)
    return (state, *selector(state))
            
    
def rollout_maker(
    data: Dict,
): 
    def maker(
        sequence: Sequence,
        encoder: Encoder,
    ):
        s_comma_a = []
        end_value = {c:-1 for c in sequence.colors}
        end_value[sequence.color] = 1
        end_value[None] = 0
        value = None

        while True:
            state = encoder(sequence, sequence.discard_pile)
            valids = sequence._valid_moves()
            try:
                # Check if state is explored
                _, _, values = data[state]

                index = randint(0, len(values) - 1)

                s_comma_a.append((state, index, sequence.current_player))
                if sequence.step(valids[index]):
                    value = lambda x: 0 if sequence.winner is None else [-1, 1][sequence.is_winner(x)]
                    break
            except KeyError:
                size = len(valids)
                data[state] = [[0] * size, [0] * size, valids]

        for state, index, player in s_comma_a:
            v = value(player)
            N, Q, _ = data[state]
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
        _, Q, _ = data[state]
        value = max(Q)
        filtered_data = [i for i, x in enumerate(Q) if x == value]
        return (valids[choice(filtered_data)],)

    return selector


__all__ = ["monte_carlo"]
