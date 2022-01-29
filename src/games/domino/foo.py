import os, json, re
from collections import defaultdict

data = defaultdict(lambda: 0)

base = 'simulations'
name_re = r'BestAccompanied-(.*)_vs_BestAccompanied-(.*).json'
files = list(os.listdir(base))

data = []
for x in files:
    p0, p1 = re.findall(name_re, x)[0]
    # if p0 == p1: continue
    with open(f'{base}/{x}', 'r') as fd:
        score = json.load(fd)
        data.append((p0, p1, score))

names = list(set([x for x, _, _ in data]))
names.sort()

# names.remove('SimpleHybrid')
# print(names)

idx = {x:i for i, x in enumerate(names)}
# print(idx)

table = [[0] * len(names) for _ in range(len(names))]

for p0, p1, score in data:
	idx0, idx1 = idx[p0], idx[p1]
	table[idx0][idx1] += score["0"]
	table[idx1][idx0] += score["1"]

for a in zip(names, table[idx['Heuristic']]): print(a)

exit(0)


# remove simpleHybrid
# idxSh = idx["SimpleHybrid"]
# table.pop(idxSh)
# for x in table:
# 	x.pop(idxSh)

# names.remove("SimpleHybrid")
for x in names: print(x)
print(' & '.join(names[0:7])) 
print(
	' \\\\\n'.join(
		' & '.join([n, *[str(x) for x in l[0:7]]])
		for n, l in zip(names, table)
	)
)
print()
print(' & '.join(names[7:])) 
print(
	' \\\\\n'.join(
		' & '.join([n, *[str(x) for x in l[7:]]])
		for n, l in zip(names, table)
	)
)


#######  Calcular los resultados finales #################
#	 for x in os.listdir(base):
#	     p0, p1 = re.findall(name_re, x)[0]
#	     with open(f'{base}/{x}', 'r') as fd:
#	         score = json.load(fd)
#	         data[p0] += score["0"]
#	         data[p1] += score["1"]
#	
#	 order = list(data.items())
#	 order.sort(key=lambda x: x[1], reverse=True)
#	
#	 for x, y in order:
#	     print(f'{x} -> {y}')
#	
#	 print("Champion:\n", '-'.join([x for x, _ in order]))