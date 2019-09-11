# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def calculatePath(startState, goalState, predecessorDict):
    path = []
    currentState = predecessorDict[goalState]
    while currentState:
        path.insert(0 ,currentState[1])
        if not currentState[0] == startState:
            currentState = predecessorDict[currentState[0]]
        else:
            currentState = None
            continue
    return path

def depthFirstSearch(problem):

    fringeList = util.Stack()
    startState = problem.getStartState()
    visited = set([startState])
    fringeList.push((startState, []))
    predecessorDict = {}
    while fringeList:
        currentNode = fringeList.pop()
        visited.add(currentNode[0])
        if problem.isGoalState(currentNode[0]):
            break
        successors = problem.getSuccessors(currentNode[0])
        for successor in successors:
            if successor[0] not in visited:
                fringeList.push((successor[0], successor[1]))
                predecessorDict[successor[0]] = (currentNode[0], successor[1])

        visited.add(currentNode[0])

    return calculatePath(startState, currentNode[0], predecessorDict)


def breadthFirstSearch(problem):
    startState = problem.getStartState()
    visited = set([startState])                                             # Visted set maintains all nodes which are visited

    fringeList = util.Queue()
    fringeList.push(startState)                                           # Storing new node in queue
    predecessorDict = {}                                                    #Stores parent and direction from it of a node.

    while fringeList:
        state = fringeList.pop()
        visited.add(state)                                               #Adding to visited as we pop from fringe list
        if (problem.isGoalState(state)):
            break

        nextStates = problem.getSuccessors(state)                        # Getting all successors
        for successor in nextStates:
            if (successor[0] in visited):
                continue
            else:
               fringeList.push(successor[0])
               predecessorDict[successor[0]] = (state, successor[1])              # Dictionary {Child: Parent, Path from parent}
    return calculatePath(startState, state, predecessorDict)

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
