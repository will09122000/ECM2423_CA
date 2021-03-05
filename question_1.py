import numpy as np
from copy import deepcopy
import time

def euclideanCost(currentState, goalState):
    modulusTotal = abs(currentState // 3 - goalState // 3) ** 2
    remainderTotal = abs(currentState % 3 - goalState % 3) ** 2
    costTotal = np.sqrt(modulusTotal + remainderTotal)
    #print(modulusTotal, remainderTotal, costTotal)
    return sum(costTotal)

def manhattanCost(currentState, goalState):
    modulusTotal = abs(currentState // 3 - goalState // 3)
    remainderTotal = abs(currentState % 3 - goalState % 3)
    costTotal = modulusTotal + remainderTotal
    #print(modulusTotal, remainderTotal, costTotal)
    return sum(costTotal)


def solvePuzzle(heuristicType, initialState, goalState):
    # Record time at which function starts.
    start = time.time()
    rows = np.array([("left", [0, 3, 6], -1),
                      ("right", [2, 5, 8], 1),
                      ("up", [0, 1, 2], -3),
                      ("down", [6, 7, 8], 3)],
                     dtype = [("move", str, 1),
                              ("position", list),
                              ("head", int)])

    dtstate = [("puzzle", list), ("parent", int),
               ("gScore", int), ("hScore", int)]

    if heuristicType == "a":
        hScore = euclideanCost(initialState, goalState)
    else:
        hScore = manhattanCost(initialState, goalState)

    state = np.array([(initialState, -1, 0, hScore)], dtstate)

    dtpriority = [("position", int),("fScore", int)]
    priority = np.array( [(0, hScore)], dtpriority)

    print("Solving Puzzle...\n")
    solved = False
    while not solved:
        # Sort grids from lowest to highest based on their fScore.
        priority = np.sort(priority, kind="mergesort",
                           order=["fScore", "position"])
        # Select the grid with the lowest fScore.
        position, fScore = priority[0]
        priority = np.delete(priority, 0, 0)
        puzzle, parent, gScore, hScore = state[position]

        nullSquare = int(np.where(puzzle == 0)[0])
        gScore += 1

        for row in rows:
            if nullSquare not in row["position"]:
                openStates = deepcopy(puzzle)
                temp = openStates[nullSquare + row["head"]]
                openStates[nullSquare + row["head"]] = openStates[nullSquare]
                openStates[nullSquare] = temp
                if not (np.all(list(state["puzzle"]) == openStates, 1)).any():
                    if heuristicType == "a":
                        hScore = euclideanCost(openStates, goalState)
                    else:
                        hScore = manhattanCost(openStates, goalState)
                    q = np.array([(openStates, position, gScore, hScore)],
                                 dtstate)
                    state = np.append(state, q, 0)
                    fScore = hScore + gScore

                    q = np.array([(len(state) - 1, fScore)], dtpriority)
                    priority = np.append(priority, q, 0)
                    if np.array_equal(openStates, goalState):
                        solved = True
                        break

    print("Puzzle has been solved!\n")
    return openStates, time.time() - start

def displayPuzzle(puzzle):
    print(str(puzzle.reshape(-1, 3, 3)).replace("  [", "").replace("[", "").replace("]", "") + "\n")


def main():
    # Define the initial and goal states of the puzzle as a numpy array.
    initialState = np.array([7, 2, 4, 5, 0, 6, 8, 3, 1])
    goalState = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])

    heuristicType = ""
    while heuristicType != "a" and heuristicType != "b":
        heuristicType = input("Pick a heuristic function:\na. Euclidean Distance \nb. Manhattan Distance\n")

    print("Initial State: \n")
    displayPuzzle(initialState)
    print("Goal State: \n")
    displayPuzzle(goalState)

    openStates, time = solvePuzzle(heuristicType, initialState, goalState)

    print("End State: \n")
    displayPuzzle(openStates)
    print("Time Taken: " + str(round(time, 2)) + " seconds")
    #print(manhattanCost(swapIndexValues(initialState), swapIndexValues(goalState)))

if __name__ == "__main__":
    main()