import random
from ..domino import DominoManager, Event

class BaseRule:
    """
        Several matches of domino are played most of the time
        on particular rules to determine the winner. This is a
        wrapper to implement and play with different rules.
    """
    def __init__(self, timeout):
        self.timeout = timeout

    def start(self, player0, player1, hand, *pieces_config):
        """
            Return id of winner team (-1 for tie)
        """
        raise NotImplementedError()


class OneGame(BaseRule):
    """
        Play one game
    """
    def start(self, player0, player1, player2, player3, hand, *pieces_config):
        env = DominoManager(timeout=self.timeout)
        players = [player0("0"), player1("1"), player2("2"), player3("3")]
        return env.run(players, 0, hand, *pieces_config)


class TwoOfThree(BaseRule):
    """
        First to win two games. Last winner start next match
    """
    def __init__(self, random_start=True, **kwargs):
        super().__init__(**kwargs)
        self.random_start = random_start

    def start(self, player0, player1, player2, player3, hand, *pieces_config):
        env = DominoManager(timeout=self.timeout)
        players = [player0("0"), player1("1"), player2("2"), player3("3")]

        cur_start = 0

        if self.random_start:
            if random.choice([False, True]):
                cur_start ^= 1

        wins = [0, 0]

        while max(wins) < 2:
            result = env.run(players, cur_start, hand, *pieces_config)

            if result != -1:
                wins[result] += 1

            if result == -1 or result != cur_start:
                # Swap players
                cur_start ^= 1

        return 0 if wins[0] > wins[1] else 1


class FirstToGain100(BaseRule):
    """
        First to team that gain 100 points. Last winner start next match
    """
    def __init__(self, random_start=True, **kwargs):
        super().__init__(**kwargs)
        self.random_start = random_start

    def update_score(self, **kwargs):
        return sum([kwargs['domino'].score(player) for player in kwargs['players']])

    def start(self, player0, player1, player2, player3, hand, *pieces_config):
        env = DominoManager(timeout=self.timeout)
        players = [player0("0"), player1("1"), player2("2"), player3("3")]

        cur_start = 0

        if self.random_start and random.choice([False, True]):
            cur_start ^= 1

        points = [0, 0]

        while max(points) < 100:
            result = env.run(players, cur_start, hand, *pieces_config, points[:])

            if result != -1:
                loser = result ^ 1
                points[result] += self.update_score(
                    players=[loser, loser + 2],
                    domino=env.domino,
                )

            if result == -1 or result != cur_start:
                # Swap players
                cur_start ^= 1

        return 0 if points[0] > points[1] else 1


class FirstDoble(FirstToGain100):
    """
        First to team that gain 100 points counting the first round doble. 
        Last winner start next match
    """
    def __init__(self, **args):
        super().__init__(**args)
        self.first_round = True

    def update_score(self, **kwargs):
        score = super().update_score(**kwargs) * (self.first_round + 1)
        self.first_round = False
        return score


class CapicuaDoble(FirstToGain100):
    """
        First to team that gain 100 points counting the capicua doble. 
        Last winner start next match
    """
    def __init__(self, **args):
        super().__init__(**args)

    def update_score(self, **kwargs):
        score = super().update_score(**kwargs)
        
        domino = kwargs['domino']
        capicua = domino.logs[-2][0] == Event.FINAL and domino.heads[0] == domino.heads[1]
        return score * (capicua + 1)
    

__all__ = ["OneGame", "TwoOfThree", "FirstToGain100", "FirstDoble", "CapicuaDoble"]
