import argparse
from module import alphazero_vs_monte_carlo, AlphaZeroModel

def main(args):
    net = AlphaZeroModel.Net()
    net.save_path = ''
    _, NN = net.load(args.path, tag=args.model, load_model=True)

    alphazero_vs_monte_carlo(
        (args.handoutsP0, args.rolloutsP0, NN),
        (args.handoutsP1, args.rolloutsP1),
        args.rep,
        args.game_config,
    )
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser("DomAIno Tournament")

    subparsers = parser.add_subparsers()

    play_parser = subparsers.add_parser('alphazero_vs_monte_carlo', help="Run a alphazero_vs_monte_carlo tournament")
    play_parser.add_argument('-n',   '--nine',            dest='game_config', action='store_const', const=[9,10], default=[6, 7], help="Double nine mode")
    play_parser.add_argument('-rep', '--repetitions',     dest='rep',         type=int, default=10, help="Numbers of plays to run")
    play_parser.add_argument('-h0',  '--handoutsPlayer0', dest='handoutsP0',  type=int, default=10, help="Numbers of handouts for AlphaZero")
    play_parser.add_argument('-h1',  '--handoutsPlayer1', dest='handoutsP1',  type=int, default=10, help="Numbers of handouts for Monte Carlo")
    play_parser.add_argument('-r0',  '--rolloutsPlayer0', dest='rolloutsP0',  type=int, default=50, help="Numbers of rollouts for Monte Carlo")
    play_parser.add_argument('-r1',  '--rolloutsPlayer1', dest='rolloutsP1',  type=int, default=50, help="Numbers of rollouts for Monte Carlo")
    play_parser.add_argument('-m',   '--model',           dest='model', help='NN\'s tag')
    play_parser.add_argument('-p',   '--path',            dest='path', default='module/training/checkpoints', help='NN\'s folder path')
    play_parser.set_defaults(command=main)

    args = parser.parse_args()

    if not hasattr(args, 'command'):
        parser.print_help()
    else:
        args.command(args)
