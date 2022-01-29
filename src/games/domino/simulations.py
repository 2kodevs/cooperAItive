from domino import get_parser
from module import PLAYERS, RULES
import json
import os


def run(player0, player1, b=None, rule='onegame', nine=True, hand='hand_out', rep=2000):
    parser = get_parser()

    extra = []
    if nine: extra.append('-n')
    if b is not None:
        player0 = f"{b}-{player0}"
        player1 = f"{b}-{player1}"

    path = f'simulations/{player0}_vs_{player1}.json'

    args = parser.parse_args(['play', '-p0', player0, '-p1', player1, '-r', rule, '-rep', str(rep), '-H', hand, '-v', *extra])

    print(f"Running {path}: ", end="")
    result = args.command(args)

    with open(path, 'w') as fd:
        json.dump(result, fd)


skip_list = ['human', 'supportive', 'rlplayer', 'montecarlo', 'alphazero', 'mc', 'a0', 'heuristic']
    

if __name__ == "__main__":
    os.makedirs('simulations/', exist_ok=True)

    for i, p0 in enumerate(PLAYERS):
        if p0.__name__.lower() in skip_list:
            continue
        for j, p1 in enumerate(PLAYERS):
            if i == j: continue
            if p1.__name__.lower() in skip_list:
                continue
            run(p0.__name__, p1.__name__, 'BestAccompanied')
