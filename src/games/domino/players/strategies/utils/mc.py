from ...player import BasePlayer
from ...player_view import PlayerView
from ....domino import Domino
from .types import *
from .game import game_hand_builder, game_data_collector, remaining_pieces


def monte_carlo(
    player: BasePlayer, 
    encoder: Encoder,
    maker: RolloutMaker,
    selector: Selector,
    handouts: int,
    rollouts: int,
) -> Tuple[State, Action] :
    # basic game information
    pieces, missing = game_data_collector(player.pieces, player.me, player.history)
    remaining = remaining_pieces(pieces, player.max_number)

    # simultations
    for _ in range(handouts):
        fixed_hands = game_hand_builder(pieces, missing, remaining, player.pieces_per_player)
        hand = lambda: [PlayerView(h) for h in fixed_hands]
        for _ in range(rollouts):
            # New Domino Game
            domino = Domino()
            domino.reset(hand, player.max_number, player.pieces_per_player)

            # Update the history
            for e, *data in player.history:
                if e.name == "MOVE":
                    move, _, head = data
                    domino.step((move, head))
                if e.name == "PASS":
                    domino.step(None)

            # Run the rollout
            maker(domino, encoder)

    # Select the player action
    state = encoder(player.pieces, player.history)
    return state, selector(state)
            