import random
import math
grid = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0]
]

start = (0, 0)
goal = (4, 4)

n = 20
iterations = 100
pa = 0.25

def generate_path():
    path = [start]
    x, y = start

    while (x, y) != goal:
        moves = [(x+1, y), (x, y+1), (x-1, y), (x, y-1)]

        valid = []
        for a, b in moves:
            if 0 <= a < 5 and 0 <= b < 5 and grid[a][b] == 0:
                valid.append((a, b))

        if not valid:
            return generate_path()

        x, y = random.choice(valid)
        if (x, y) not in path:
            path.append((x, y))

    return path


def fitness(path):
    cost = 0

    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i+1]

        cost += math.sqrt((x2-x1)**2 + (y2-y1)**2)

    return cost

nests = [generate_path() for _ in range(n)]

best = min(nests, key=fitness)

for iteration in range(iterations):

    for i in range(n):

        new_path = generate_path()

        if fitness(new_path) < fitness(nests[i]):
            nests[i] = new_path

    nests.sort(key=fitness)

    for i in range(int(pa * n)):
        nests[-(i+1)] = generate_path()

    current_best = min(nests, key=fitness)

    if fitness(current_best) < fitness(best):
        best = current_best

print("Best path:", best)
print("Minimum path cost:", fitness(best))
