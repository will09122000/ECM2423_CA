import numpy as np
from copy import deepcopy
 
def swapIndexValues(state):
    """
    Swaps an array so each index is placed at its current value.

        Parameters:
            state: An integer array
        
        Returns:
            swapped: An integer array with the swapped index, values.
    """
    swapped = np.array(range(9))
    for index, value in enumerate(state):
        swapped[value] = index
    return swapped

def misplacedCost(currentState, goalState):
    cost = 0
    for currentNum, goalNum in zip(currentState, goalState):
        if currentNum != goalNum:
            cost += 1
    return cost

def manhattanCost(currentState, goalState):
    modulusTotal = abs(currentState // 3 - goalState // 3)
    remainderTotal = abs(currentState % 3 - goalState % 3)
    costTotal = modulusTotal + remainderTotal
    return sum(costTotal[1:9])


def solvePuzzle(heuristicType, initialState, goalState):
    steps = np.array([('up', [0, 1, 2], -3),('down', [6, 7, 8],  3),('left', [0, 3, 6], -1),('right', [2, 5, 8], 1)],
                dtype =  [('move', str, 1),('position', list),('head', int)])

    dtstate = [('puzzle', list),('parent', int),('gScore',  int),('hScore',  int)]

    parent = -1
    gScore = 0
    if heuristicType == "a":
        hScore = misplacedCost(swapIndexValues(initialState), swapIndexValues(goalState))
    else:
        hScore = manhattanCost(swapIndexValues(initialState), swapIndexValues(goalState))
    state = np.array([(initialState, parent, gScore, hScore)], dtstate)
    
    
    solved = False

    dtpriority = [('position', int),('fScore', int)]
    priority = np.array( [(0, hScore)], dtpriority)

    print("Solving Puzzle...\n")
    while not solved:
        priority = np.sort(priority, kind='mergesort', order=['fScore', 'position'])
        position, fScore = priority[0]
        priority = np.delete(priority, 0, 0)
        puzzle, parent, gScore, hScore = state[position]
        """
        puzzle = np.array(stateTest[position]["puzzle"])
        parent = stateTest[position]["parent"]
        gScore = stateTest[position]["gScore"]
        hScore = stateTest[position]["hScore"]
        """

        nullSquare = int(np.where(puzzle == 0)[0])
        gScore += 1

        for step in steps:
            if nullSquare not in step['position']:
                openStates = deepcopy(puzzle)
                temp = openStates[nullSquare + step['head']]
                openStates[nullSquare + step['head']] = openStates[nullSquare]
                openStates[nullSquare] = temp
                if not (np.all(list(state['puzzle']) == openStates, 1)).any():
                    if heuristicType == "a":
                        hScore = misplacedCost(swapIndexValues(initialState), swapIndexValues(goalState))
                    else:
                        hScore = manhattanCost(swapIndexValues(initialState), swapIndexValues(goalState))
                    q = np.array([(openStates, position, gScore, hScore)], dtstate)
                    state = np.append(state, q, 0)
                    fScore = gScore + hScore

                    q = np.array([(len(state) - 1, fScore)], dtpriority)
                    priority = np.append(priority, q, 0)
                    if np.array_equal(openStates, goalState):
                        solved = True

    print("Puzzle has been solved!\n")
    return state, priority

def displayPuzzle(puzzle):
    print(str(puzzle.reshape(-1, 3, 3)).replace("  [", "").replace("[", "").replace("]", "") + "\n")


def main():
    initialState = np.array([7, 2, 4, 5, 0, 6, 8, 3, 1])
    goalState = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    print("Initial State: \n")
    displayPuzzle(initialState)
    print("Goal State: \n")
    displayPuzzle(goalState)


    heuristicType = ""
    while heuristicType != "a" and heuristicType != "b":
        heuristicType = input('Pick a heuristic function:\na. Misplaced Tiles \nb. Manhattan Distance\n')

    state, visited = solvePuzzle(heuristicType, initialState, goalState)

if __name__ == "__main__":
    main()