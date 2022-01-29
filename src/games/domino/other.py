import json

with open('last_exp.json', 'r') as fd:
    data = json.load(fd)

class Data:
    def __init__(self) -> None:
        self.g = {}
        self.g['a0'] = 0
        self.g['h'] =  0
        self.v = {}
        self.v['a0'] = [0, 0]
        self.v['h'] =  [0, 0]
        self.p = {}
        self.p['a0'] = [0, 0]
        self.p['h'] =  [0, 0]

    def add(self, game, _, a0, h0, a1, h1):
        players = game.split('_vs_')
        self.g['a0'] += 1
        self.g['h'] += 1
        for p, (v0, v1) in zip(players, ((a0, a1), (h0, h1))):
            self.v[p][0] += v0 * v0
            self.v[p][1] += v1 * v1
            self.p[p][0] += v0
            self.p[p][1] += v1

    def add_custom(self, game, _, a0, h0, a1, h1, idx):
        players = game.split('_vs_')
        self.g[players[idx]] += 1
        p, (v0, v1) = players[idx], ((a0, a1), (h0, h1))[idx]
        self.v[p][0] += v0 * v0
        self.v[p][1] += v1 * v1
        self.p[p][0] += v0
        self.p[p][1] += v1

    def after_all(self):
        # promedio
        self.p['a0'][0] /= max(1, self.g['a0'])
        self.p['a0'][1] /= max(1, self.g['a0'])
        self.p['h'][0]  /= max(1, self.g['h'])
        self.p['h'][1]  /= max(1, self.g['h'])

        # varianza
        self.v['a0'][0] /= max(1, self.g['a0'])
        self.v['a0'][1] /= max(1, self.g['a0'])
        self.v['h'][0]  /= max(1, self.g['h'])
        self.v['h'][1]  /= max(1, self.g['h'])

        self.v['a0'][0] -= self.p['a0'][0] ** 2
        self.v['a0'][1] -= self.p['a0'][1] ** 2
        self.v['h'][0]  -= self.p['h'][0] ** 2
        self.v['h'][1]  -= self.p['h'][1] ** 2

    def log(self, label):
        print(
            f'{label}:\n'
             "  Promedio:\n"
            f"    AlphaZero0: {self.p['a0'][0]}\n"
            f"    AlphaZero1: {self.p['a0'][1]}\n"
            f"    Heuristic0: {self.p['h'][0]}\n"
            f"    Heuristic1: {self.p['h'][1]}\n"
             "  Varianza:\n"
            f"    AlphaZero0: {self.v['a0'][0]}\n"
            f"    AlphaZero1: {self.v['a0'][1]}\n"
            f"    Heuristic0: {self.v['h'][0]}\n"
            f"    Heuristic1: {self.v['h'][1]}\n"
        )


general = Data()
win = Data()
lose = Data()

for x in data:
    general.add(*x)
    if int(x[1]) != -1:
        win.add_custom(*x, int(x[1]))
        lose.add_custom(*x, 1 - int(x[1]))

general.after_all()
win.after_all()
lose.after_all()

general.log("Todos los juegos")
win.log("En juegos ganados")
lose.log("En juegos perdidos")
