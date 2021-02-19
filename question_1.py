import numpy as np

def misplacedCost(currentState, goalState):
    cost = 0
    for currentNum, goalNum in zip(currentState, goalState):
        if currentNum != goalNum:
            cost += 1
    return cost

def coordinates(state):
    coordinates = np.array(range(9))
    for x, y in enumerate(state):
        coordinates[y] = x
    return coordinates

def manhattanCost(currentState, goalState):
    coordinates = np.array(range(9))
    for x, y in enumerate(currentState):
        coordinates[y] = x
    cost = abs(currentState // 3 - goalState // 3) + abs(currentState % 3 - goalState % 3)
    return sum(cost[1:])


initialState = [7, 2, 4, 5, 0, 6, 8, 3, 1]
goalState = [0, 1, 2, 3, 4, 5, 6, 7, 8]

function_type = input('Pick a heuristic function:\na. Misplaced Tiles \nb. Manhattan Distance\n')

print(manhattanCost(coordinates(initialState), coordinates(goalState)))
