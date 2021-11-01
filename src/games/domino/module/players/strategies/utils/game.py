import random
from collections import defaultdict

from .types import Domino, Event


class EdmondsKarp:
    class edge:
        def __init__(self, start, end, f, cap):
            self.start = start
            self.end   = end
            self.cap   = cap
            self.f     = f

    inf = float('inf')

    def __init__(self, s, t):
        self.source = s
        self.sink   = t
        self.G = [[] for _ in range(t + 1)]
        self.E = []

    def add_edge(self, start, end, cap):
        self.G[start].append(len(self.E))
        self.E.append(self.edge(start, end, cap, cap))
        self.G[end].append(len(self.E))
        self.E.append(self.edge(end, start, 0, cap))
    
    def solve(self):
        def bfs():
            dad = [-1] * (self.sink + 1)
            e   = [-1] * (self.sink + 1)
            dad[self.source] = -2

            q = []
            q.append(self.source)
            while q:
                v = q[0]
                q.pop(0)

                if v == self.sink: break

                for i in range(len(self.G[v])):
                    idx = self.G[v][i]
                    next = self.E[idx].end

                    if dad[next] == -1 and self.E[idx].f:
                        dad[next] = v
                        e[next] = idx
                        q.append(next)
            if dad[self.sink] == -1:
                return 0
            
            cur, flow = self.sink, self.inf
            while cur != self.source:
                idx = e[cur]
                flow = min(flow, self.E[idx].f)
                cur = dad[cur]
            cur = self.sink
            while cur != self.source:
                idx = e[cur]
                self.E[idx].f -= flow
                self.E[idx ^ 1].f += flow
                cur = dad[cur]
            return flow

        flow, f = 0, 1
        while f:
            f = bfs()
            flow += f
        return flow

    
def game_data_collector(current_hand, player_id, history):
    pieces = [[], [], [], []]
    pieces[player_id].extend(current_hand)
    missing = [[], [], [], []]

    heads = [-1, -1]
    empty = [-1, -1]
    for event, *data in history:
        if event.name == 'MOVE':
            id, move, head = data 
            pieces[id].append(move)
            if heads == empty:
                heads = list(move)
            else:
                heads[head] = move[move[0] == heads[head]]
        elif event.name == 'PASS':
            id = data[0]
            missing[id].extend(heads)
    return pieces, [set(s) for s in missing]


def remaining_pieces(pieces, max_number=7):
    remaining = []
    taken = set()
    for player in pieces:
        for a, b in player:
            taken.add((min(a, b), max(a, b)))
    for i in range(max_number + 1):
        for j in range(i, max_number + 1):
            if (i,j) not in taken:
                remaining.append((i, j))
    return remaining

    
def game_hand_builder(pieces, missing, remaining, number_of_pieces=7):
    pieces = [[x for x in p] for p in pieces]   

    random.shuffle(remaining)
    source, sink = 0, 4 + len(remaining) + 1
    flow = EdmondsKarp(source, sink)

    for player in range(1, 5):
        s = missing[player - 1]
        for i, piece in enumerate(remaining):
            for p in piece:
                if p in s:
                    break
            else:
                flow.add_edge(player, i + 5, 1)
    last_assigment_edge = len(flow.E)
    for player in range(1, 5):
        cap = number_of_pieces - len(pieces[player - 1]) 
        flow.add_edge(source, player, cap)
    for i in range(len(remaining)):
        flow.add_edge(i + 5, sink, 1)

    rep = number_of_pieces * 4 - sum([len(x) for x in pieces])
    assert flow.solve() == rep, 'Impossible to find a pieces assigment'

    for i in range(0, last_assigment_edge, 2):
        edge = flow.E[i]
        if not edge.f:
            pieces[edge.start - 1].append(remaining[edge.end - 5])
    
    return pieces


def calc_colab(domino: Domino, player):
    partner = player ^ 2
    prev = (partner + 3) % 4

    f = 0
    pegue = 0
    tranque = 0
    p = domino.winner == (player & 1)
    points = sum([x + y for x, y in domino.players[player].remaining])
    all_pieces = sum([x + y for x, y in domino.players[player].pieces])
    passs = 0
    rep = 0
    partner_pieces = domino.players[partner].pieces[:]
    partner_moves, prev_moves = 0, 0

    heads = None
    def valid(piece):
        return heads[0] in piece or heads[1] in piece

    prev_data = defaultdict(lambda: 0)
    if domino.logs[0][0] is Event.NEW_GAME:
        domino.logs.pop(0)
    for (e, *details) in domino.logs:
        if e is Event.PASS:
            passs += details[0] == partner
        elif e is Event.MOVE:
            playerId, piece, head = details
            if playerId == prev:
                prev_moves += 1
                if heads is None:
                    prev_data[piece[0]] = 1
                    prev_data[piece[1]] = 1
                else:
                    prev_data[piece[piece[0] == heads[head]]] += 1
            elif playerId == partner:
                partner_moves += 1
                if heads is None:
                    f = 1
                else:
                    f += len([p for p in partner_pieces if valid(p)])
                    partner_pieces.remove(piece)
            if heads is None:
                heads = list(piece)
            else:
                heads[head] = piece[piece[0] == heads[head]]
        elif e is Event.OVER:
            tranque = 1
        elif e is Event.FINAL:
            pegue = details[0] == partner
 
    partner_turns = partner_moves + passs
    f /= 2 * partner_turns
    rep = sum(prev_data.values()) - len(prev_data)
    passs /= partner_turns
    rep /= prev_moves
    end_value = [0, 0, 0]
    end_value[player & 1] = 1 
    end_value[1 - (player & 1)] = -1 
    v = end_value[domino.winner]

    return (f + pegue + tranque*p*points/all_pieces - (passs + rep) + v) / 6


__all__ = [
    "game_data_collector", 
    "game_hand_builder", 
    "remaining_pieces",
    "calc_colab",
]
