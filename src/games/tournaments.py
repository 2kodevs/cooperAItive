import argparse
from domino import alphazero_vs_monte_carlo

def main():
    parser = argparse.ArgumentParser("DomAIno Tournament")

    subparsers = parser.add_subparsers()

    play_parser = subparsers.add_parser('alphazero_vs_monte_carlo', help="Run a alphazero_vs_monte_carlo tournament")
    play_parser.add_argument('-n',   '--nine',            dest='game_config', action='store_const', const=[9,10], default=[6, 7])
    play_parser.add_argument('-rep', '--repetitions',     dest='rep',         type=int, default=10)
    play_parser.add_argument('-h0',  '--handoutsPlayer0', dest='handoutsP0',  type=int, default=10)
    play_parser.add_argument('-h1',  '--handoutsPlayer1', dest='handoutsP1',  type=int, default=10)
    play_parser.add_argument('-r0',  '--rolloutsPlayer0', dest='rolloutsP0',  type=int, default=50)
    play_parser.add_argument('-r1',  '--rolloutsPlayer1', dest='rolloutsP1',  type=int, default=50)
    play_parser.add_argument('-m',   '--model',           dest='model', help='NN\'s tag')
    play_parser.add_argument('-p',   '--path',            dest='path', default='domino/training/checkpoints', help='NN\'s folder path')
    play_parser.set_defaults(command=alphazero_vs_monte_carlo)

    args = parser.parse_args()

    if not hasattr(args, 'command'):
        parser.print_help()
    else:
        args.command(args)

if __name__ == '__main__':
    main()