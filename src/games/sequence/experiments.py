from sequence import get_parser
from utils import prepare_player as get_player
import argparse, json, time
from module import get_hand, SequenceManager, game_utils


hand_out = get_hand("handout")


def experiment_heuristic_vs_mcts(args):
    parser = get_parser()

    player = ['mc', args.h, args.r]
    
    print(
        "Experiment #1 ----------------\n"
        "  Runs:\n"
        "    heuristic_vs_mcts:",
        end=" ", flush=True,
    )
    parsed_args = parser.parse_args(['play', '-p0', 'heuristic', '-p1', *player, "-v", '-rep', args.rep])
    x = parsed_args.command(parsed_args)

    print("    mcts_vs_heuristic:", end=" ", flush=True)
    parsed_args = parser.parse_args(['play', '-p1', 'heuristic', '-p0', *player, "-v", '-rep', args.rep])
    y = parsed_args.command(parsed_args)

    player_score = x[1] + y[0]
    heuristic_score = x[0] + y[1]
    print(
        "  Scores:\n"
        f"    Heuristic: {heuristic_score}\n"
        f"    MCTS: {player_score}\n"
        "------------------------------"
    )
    return "MCTS", player_score


def experiment_heuristic_vs_a0(args):
    parser = get_parser()

    player = ['a0', args.h, args.r, args.nn, '0']
    
    print(
        "Experiment #2 ----------------\n"
        "  Runs:\n"
        "    heuristic_vs_alphazero:",
        end=" ", flush=True,
    )
    parsed_args = parser.parse_args(['play', '-p0', 'heuristic', '-p1', *player, "-v", '-rep', args.rep])
    x = parsed_args.command(parsed_args)

    print("    alphazero_vs_heuristic:", end=" ", flush=True)
    parsed_args = parser.parse_args(['play', '-p1', 'heuristic', '-p0', *player, "-v", '-rep', args.rep])
    y = parsed_args.command(parsed_args)

    player_score = x[1] + y[0]
    heuristic_score = x[0] + y[1]
    print(
        "  Scores:\n"
        f"    Heuristic: {heuristic_score}\n"
        f"    Alpha Zero: {player_score}\n"
        "------------------------------"
    )
    return "Alpha Zero", player_score


def experiment_heuristic_vs_a0coop(args):
    parser = get_parser()

    player = ['a0', args.h, args.r, args.nn, "5"]
    
    print(
        "Experiment #3 ----------------\n"
        "  Runs:\n"
        "    heuristic_vs_alphazero_coop:",
        end=" ", flush=True,
    )
    parsed_args = parser.parse_args(['play', '-p0', 'heuristic', '-p1', *player, "-v", '-rep', args.rep])
    x = parsed_args.command(parsed_args)

    print("    alphazero_coop_vs_heuristic:", end=" ", flush=True)
    parsed_args = parser.parse_args(['play', '-p1', 'heuristic', '-p0', *player, "-v", '-rep', args.rep])
    y = parsed_args.command(parsed_args)

    player_score = x[1] + y[0]
    heuristic_score = x[0] + y[1]
    print(
        "  Scores:\n"
        f"    Heuristic: {heuristic_score}\n"
        f"    Alpha Zero Coop: {player_score}\n"
        "------------------------------"
    )
    return "Alpha Zero Coop", player_score


def test_handouts_vs_rollout(args):
    parser = get_parser()

    print(
        "Test #1 ----------------------\n"
        "  Running handouts_vs_rollouts:"
    )
    total = 1000
    handouts = ["1", "2", "5", "10", "20", "50", "100", "200", "500", "1000"]

    score = [0] * 10
    for i, h0 in enumerate(handouts):
        r0 = str(total // int(h0))
        for j, h1 in enumerate(handouts):
            if j == i: continue
            r1 = str(total // int(h1))
            print(f"    {h0}_vs_{h1}: ", end="", flush=True)
            args = parser.parse_args(['play', '-p0', 'mc', h0, r0, '-p1', "mc", h1, r1, "-v", '-rep', '30'])
            X = args.command(args)
            score[i] += X[0]
            score[j] += X[1]

    tup = list(zip(handouts, score))
    tup.sort(key=lambda x: x[1], reverse=True)
    print("  Results (handout -> wins):")
    for x, y in tup:
        print(f"    {x} -> {y}")
    print("------------------------------")


def test_colab(args):
    heuristic = get_player(["heuristic"])
    A0 = get_player(["a0", args.h, args.r, args.nn, 15])

    data = []
    print("sefolmo")
    now = time.time()
    for _ in range(int(args.rep)):
        manager = SequenceManager()
        players = [heuristic, A0, heuristic, A0]
        result = manager.run(hand_out, players, "0101", 7, 2)
        data.append(('h_vs_a0', result, *[game_utils.calc_colab(manager.seq, i) for i in range(4)]))
    print("round tu fait")
    print(f"duration: ${time.time() - now}")
    for _ in range(int(args.rep)):
        manager = SequenceManager()
        players = [A0, heuristic, A0, heuristic]
        result = manager.run(hand_out, players, "0101", 7, 2)
        data.append(('a0_vs_h', result, *[game_utils.calc_colab(manager.seq, i) for i in range(4)])) 
    print(data)

    with open('last_exp.json', 'w') as fd:
        json.dump(data, fd, indent=4)


def main(args):
    # Run tests
    for test in args.tests:
        test(args)

    # Run experiments
    scores = []
    for exp in args.experiments:
        scores.append(exp(args))
    print(
        "Final scores -----------------\n ",
        "\n  ".join(f'{label}: {score}' for label, score in scores),
        "\n------------------------------",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DomAIno-Experiments")

    parser.add_argument(
        '-e1', '--exp1', dest="experiments", 
        action='append_const', const=experiment_heuristic_vs_mcts, 
        help="Heuristic vs MCTS experiment",
    )
    parser.add_argument(
        '-e2', '--exp2', dest="experiments", 
        action='append_const', const=experiment_heuristic_vs_a0,
        help="Heuristic vs Alpha Zero experiment",
    )
    parser.add_argument(
        '-e3', '--exp3', dest="experiments", 
        action='append_const', const=experiment_heuristic_vs_a0coop,
        help="Heuristic vs Alpha Zero with coop experiment",
    )
    parser.add_argument(
        '-t1', '--test1', dest="tests", 
        action='append_const', const=test_handouts_vs_rollout,
        help="Handouts vs Rollouts test",
    )
    parser.add_argument(
        '-t2', '--test2', dest="tests", 
        action='append_const', const=test_colab,
        help="Colab calculation",
    )
    parser.add_argument(
        '-nn', '--network', dest="nn", 
        default='module/training/checkpoints/experimet_player.ckpt', 
        type=str, help="Neural Network path",
    )
    parser.add_argument(
        '-H', '--handouts', dest="h", 
        default='100', type=str, help="Number of handouts",
    )
    parser.add_argument(
        '-r', '--rollouts', dest="r", 
        default='10', type=str, help="Number of rollouts",
    )
    parser.add_argument(
        '-rep', '--repetitions', dest="rep", 
        default='50', type=str, help="Number of repetitions",
    )
    parser.set_defaults(experiments=[], tests=[], command=main)

    args = parser.parse_args()

    if not hasattr(args, 'command'):
        parser.print_help()
    else:
        args.command(args)