from domino import get_parser
import argparse


def experiment_heuristic_vs_mcts():
    parser = get_parser()

    player = ['mc', '100', '10']
    
    print(
        "Experiment #1 ----------------\n"
        "  Runs:\n"
        "    heuristic_vs_mcts:",
        end=" ", flush=True,
    )
    args = parser.parse_args(['play', '-p0', 'heuristic', '-p1', *player, "-v", '-rep', '100'])
    x = args.command(args)

    print("    mcts_vs_heuristic:", end=" ", flush=True)
    args = parser.parse_args(['play', '-p1', 'heuristic', '-p0', *player, "-v", '-rep', '100'])
    y = args.command(args)

    player_score = x[1] + y[0]
    heuristic_score = x[0] + y[1]
    print(
        "  Scores:\n"
        f"    Heuristic: {heuristic_score}\n"
        f"    MCTS: {player_score}\n"
        "------------------------------"
    )
    return "MCTS", player_score


def experiment_heuristic_vs_a0():
    parser = get_parser()

    player = ['a0', '100', '10', 'module/training/checkpoints/experimet_player.ckpt', '0']
    
    print(
        "Experiment #2 ----------------\n"
        "  Runs:\n"
        "    heuristic_vs_alphazero:",
        end=" ", flush=True,
    )
    args = parser.parse_args(['play', '-p0', 'heuristic', '-p1', *player, "-v", '-rep', '100'])
    x = args.command(args)

    print("    alphazero_vs_heuristic:", end=" ", flush=True)
    args = parser.parse_args(['play', '-p1', 'heuristic', '-p0', *player, "-v", '-rep', '100'])
    y = args.command(args)

    player_score = x[1] + y[0]
    heuristic_score = x[0] + y[1]
    print(
        "  Scores:\n"
        f"    Heuristic: {heuristic_score}\n"
        f"    Alpha Zero: {player_score}\n"
        "------------------------------"
    )
    return "Alpha Zero", player_score


def experiment_heuristic_vs_a0coop():
    parser = get_parser()

    player = ['a0', '100', '10', 'module/training/checkpoints/experimet_player.ckpt', "13"]
    
    print(
        "Experiment #3 ----------------\n"
        "  Runs:\n"
        "    heuristic_vs_alphazero_coop:",
        end=" ", flush=True,
    )
    args = parser.parse_args(['play', '-p0', 'heuristic', '-p1', *player, "-v", '-rep', '100'])
    x = args.command(args)

    print("    alphazero_coop_vs_heuristic:", end=" ", flush=True)
    args = parser.parse_args(['play', '-p1', 'heuristic', '-p0', *player, "-v", '-rep', '100'])
    y = args.command(args)

    player_score = x[1] + y[0]
    heuristic_score = x[0] + y[1]
    print(
        "  Scores:\n"
        f"    Heuristic: {heuristic_score}\n"
        f"    Alpha Zero Coop: {player_score}\n"
        "------------------------------"
    )
    return "Alpha Zero Coop", player_score


def test_handouts_vs_rollout():
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


def main(args):
    # Run tests
    for test in args.tests:
        test()

    # Run experiments
    scores = []
    for exp in args.experiments:
        scores.append(exp())
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
    parser.set_defaults(experiments=[], tests=[], command=main)

    args = parser.parse_args()

    if not hasattr(args, 'command'):
        parser.print_help()
    else:
        args.command(args)
