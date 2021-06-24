import random
from ..domino import DominoManager, Event

class BaseRule:
    """
        Several matches of domino are played most of the time
        on particular rules to determine the winner. This is a
        wrapper to implement and play with different rules.
    """
    def run(self, players):
        """
            Return id of winner team (-1 for tie)
        """
        raise NotImplementedError()


class OneGame:
    """
        Play one game
    """
    def start(self, player0, player1, hand, *pieces_config):
        env = DominoManager()
        players = [player0("0"), player1("1"), player0("2"), player1("3")]
        return env.run(players, hand, *pieces_config)


class TwoOfThree:
    """
        First to win two games. Last winner start next match
    """
    def __init__(self, random_start=True):
        self.random_start = random_start

    def start(self, player0, player1, hand, *pieces_config):
        env = DominoManager()
        players = [player0("0"), player1("1"), player0("2"), player1("3")]

        cur_start = 0

        if self.random_start:
            if random.choice([False, True]):
                cur_start ^= 1
                players[0], players[1] = players[1], players[0]
                players[2], players[3] = players[3], players[2]

        wins = [0, 0]

        while max(wins) < 2:
            result = env.run(players, hand, *pieces_config)

            if result != -1:
                wins[result ^ cur_start] += 1

            if result == -1 or result != cur_start:
                # Swap players
                cur_start ^= 1
                players[0], players[1] = players[1], players[0]
                players[2], players[3] = players[3], players[2]

        return 0 if wins[0] > wins[1] else 1


class FirstToGain100:
    """
        First to team that gain 100 points. Last winner start next match
    """
    def __init__(self, random_start=True):
        self.random_start = random_start

    def update_score(self, **kwargs):
        return sum([player.score() for player in kwargs['players']])

    def start(self, player0, player1, hand, *pieces_config):
        env = DominoManager()
        players = [player0("0"), player1("1"), player0("2"), player1("3")]

        cur_start = 0

        if self.random_start:
            if random.choice([False, True]):
                cur_start ^= 1
                players[0], players[1] = players[1], players[0]
                players[2], players[3] = players[3], players[2]

        points = [0, 0]

        while max(points) < 100:
            result = env.run(players, hand, *pieces_config)

            if result != -1:
                loser = result ^ 1
                points[result ^ cur_start] += self.update_score(
                    players=[players[loser], players[loser + 2]],
                    domino=env.domino,
                )

            if result == -1 or result != cur_start:
                # Swap players
                cur_start ^= 1
                players[0], players[1] = players[1], players[0]
                players[2], players[3] = players[3], players[2]

        return 0 if points[0] > points[1] else 1


class FirstDoble(FirstToGain100):
    """
        First to team that gain 100 points counting the first round doble. 
        Last winner start next match
    """
    def __init__(self, *args):
        super().__init__(*args)
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
    def __init__(self, *args):
        super().__init__(*args)

    def update_score(self, **kwargs):
        score = super().update_score(**kwargs)
        
        domino = kwargs['domino']
        capicua = domino.logs[-2][0] == Event.FINAL and domino.heads[0] == domino.heads[1]
        return score * (capicua + 1)
    