def count_min(player, piece):
    cant = [0, 0]
    for item in player.pieces:
        cant[0] += (piece[0] in item)
        cant[1] += (piece[1] in item)
    val = min(cant)
    return val, cant.index(val)


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
            
            cur = self.sink, flow = self.inf
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
            move, id, head = data 
            pieces[id].append(move)
            if heads == empty:
                heads = move
            else:
                heads[head] = move[move[0] == heads[head]]
        elif event.name == 'PASS':
            id = data[0]
            missing[id].extend(heads)
    return pieces, missing
    