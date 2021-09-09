from .players import MonteCarlo, AlphaZero, get_hand
from .domino import DominoManager


def runner(data, rep, output, game_config, hand='hand_out'):
    [(p0, *args0), (p1, *args1)] = data
    p0name = p0.__name__
    p1name = p1.__name__
    key = f'{p0name}_vs_{p1name}'
    d = output[key] = output.get(key, {-1:0, 0:0, 1:0})
    for _ in range(rep):
        manager = DominoManager()
        winner = manager.run([
                p0(f'{p0name}0', *args0),
                p1(f'{p0name}1', *args1),
                p0(f'{p0name}2', *args0),
                p1(f'{p0name}3', *args1),
            ], 
            get_hand(hand),
            *game_config,
        )
        d[winner] += 1


def alphazero_vs_monte_carlo(alphaArgs, mcArgs, rep, game_config):
    output = {}
    data0 = (AlphaZero, *alphaArgs)
    data1 = (MonteCarlo, *mcArgs)
    runner([data0, data1], rep, output, game_config)
    runner([data1, data0], rep, output, game_config)
    print(output)


__all__ = ["alphazero_vs_monte_carlo"]
