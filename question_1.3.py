from copy import deepcopy
import numpy as np
import time

def euclideanCost(currentState, goalState):
    """
    Calculates the current Euclidean Distance heuristic cost.

    Keyword arguments:
    currentState -- the current state of the puzzle
    goalState -- the goal state of the puzzle
    """

    modulusTotal = abs(currentState // 3 - goalState // 3) ** 2
    remainderTotal = abs(currentState % 3 - goalState % 3) ** 2
    costTotal = np.sqrt(modulusTotal + remainderTotal)
    return sum(costTotal)

def manhattanCost(currentState, goalState):
    """
    Calculates the current Manhattan Distance heuristic cost.

    Keyword arguments:
    currentState -- the current state of the puzzle
    goalState -- the goal state of the puzzle
    """

    modulusTotal = abs(currentState // 3 - goalState // 3)
    remainderTotal = abs(currentState % 3 - goalState % 3)
    costTotal = modulusTotal + remainderTotal
    return sum(costTotal)


def solvePuzzle(heuristicType, initialState, goalState):
    """
    Solves the 8-puzzle problem using the A* algorithm.

    Keyword arguments:
    heuristicType -- the type of heuristic to be used (Euclidean or Manhattan)
    currentState -- the current state of the puzzle
    goalState -- the goal state of the puzzle
    """

    # Record time at which function starts
    start = time.time()

    # Check if the goals are already equal
    if np.array_equal(initialState, goalState):
        return goalState, time.time() - start

    # Create numpy array templates
    rows = np.array([("left", [0, 3, 6], -1),
                      ("right", [2, 5, 8], 1),
                      ("up", [0, 1, 2], -3),
                      ("down", [6, 7, 8], 3)],
                     dtype = [("move", str, 1),
                              ("position", list),
                              ("head", int)])
    puzzleState = [("puzzle", list), ("parent", int),
                   ("gScore", int), ("hScore", int)]

    # Pick a heuristic
    if heuristicType == "a":
        hScore = euclideanCost(initialState, goalState)
    else:
        hScore = manhattanCost(initialState, goalState)

    state = np.array([(initialState, -1, 0, hScore)], puzzleState)

    rank = np.array( [(0, hScore)], [("position", int),("fScore", int)])

    print("Solving Puzzle...\n")
    solved = False
    while not solved:
        # Sort grids from lowest to highest based on their fScore
        rank = np.sort(rank, kind="mergesort",
                           order=["fScore", "position"])
        # Select the grid with the lowest fScore
        position, fScore = rank[0]
        rank = np.delete(rank, 0, 0)
        puzzle, parent, gScore, hScore = state[position]

        # Find the null square and increment the gScore
        nullSquare = int(np.where(puzzle == 0)[0])
        gScore += 1

        # Iterate through each square that isn't the null square
        for row in rows:
            if not nullSquare in row["position"]:
                # Create copy of current state of puzzle
                openStates = deepcopy(puzzle)
                # Swap values using 'temp' variable
                temp = openStates[nullSquare + row["head"]]
                openStates[nullSquare + row["head"]] = openStates[nullSquare]
                openStates[nullSquare] = temp

                # If a path has yet to be explored
                if not (np.all(list(state["puzzle"]) == openStates, 1)).any():
                    # Pick a heuristic.
                    if heuristicType == "a":
                        hScore = euclideanCost(openStates, goalState)
                    else:
                        hScore = manhattanCost(openStates, goalState)

                    # Create the next queue and it to the state numpy array
                    queue = np.array([(openStates, position, gScore, hScore)],
                                     puzzleState)
                    state = np.append(state, queue, 0)

                    # Calculate total cost
                    fScore = hScore + gScore

                    queue = np.array([(len(state) - 1, fScore)],
                                     [("position", int),("fScore", int)])
                    rank = np.append(rank, queue, 0)

                    # Check if the puzzle has been solved
                    if np.array_equal(openStates, goalState):
                        solved = True
                        break

    print("Puzzle has been solved!\n")
    return openStates, time.time() - start

def displayPuzzle(state):
    """Displays the puzzle array in a way that is easier to visualize."""

    print(str(state.reshape(-1, 3, 3))
                    .replace("  [", "")
                    .replace("[", "")
                    .replace("]", "") + "\n")

def puzzleConfig(stateType):
    """
    Prompts the user to pick unique numbers 0 to 8 for either the start or the
    goal state determined by the function parameter 'stateType'.
    """

    print("\nPick numbers for the " + stateType + " state, 0 represents the empty gap.")
    state = np.zeros(9, dtype=int)
    displayPuzzle(state)
    # Iterate through 0-filled array
    for x in np.nditer(state, op_flags=['readwrite']):
        validInput = False
        inputNumber = int(input("number: "))
        # Only add the number if it's valid
        while not validInput:
            if (inputNumber > -1 and inputNumber < 9) and \
            (inputNumber not in state or inputNumber == 0):
                validInput = True
            else:
                print("Number needs to be between 0 and 8 and can only be used once.")
                displayPuzzle(state)
                inputNumber = int(input(stateType + " number: "))
        x[...] = inputNumber
        displayPuzzle(state)
    return state

def main():
    """
    The main function to solve the 8-puzzle problem with a starting goal
    determined by the user and a choice of two heuristics to be used in an A*
    algorithm.
    """

    # User defined initial and goal states of the puzzle as numpy arrays
    initialState = puzzleConfig("Start")
    goalState = puzzleConfig("Goal")

    # Asks the user to pick a heuristic
    heuristicType = ""
    while heuristicType != "a" and heuristicType != "b":
        heuristicType = input("Pick a heuristic function:\na. Euclidean Distance \nb. Manhattan Distance\n")

    # Display the initial and goal states of the puzzle
    print("Initial State: \n")
    displayPuzzle(initialState)
    print("Goal State: \n")
    displayPuzzle(goalState)

    # Solves the puzzle
    openStates, time = solvePuzzle(heuristicType, initialState, goalState)

    # Displays the end state (should be identical to the goal state) and
    # displays the time it took to solve the puzzle
    print("End State: \n")
    displayPuzzle(openStates)
    print("Time Taken: " + str(round(time, 4)) + " seconds")

if __name__ == "__main__":
    main()
