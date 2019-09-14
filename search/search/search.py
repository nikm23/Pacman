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
    return searchAlgorithm(problem, fringeList)

def breadthFirstSearch(problem):
    fringeList = util.Queue()
    return searchAlgorithm(problem, fringeList)

def searchAlgorithm(problem, fringeList):
    startState = problem.getStartState()
    visited = []
    fringeList.push((startState, []))                                 # Storing new node in queue/stack

    while not fringeList.isEmpty():
        currentNode, pathtillnow = fringeList.pop()
        visited.append(currentNode)
        if problem.isGoalState(currentNode):
            return pathtillnow
        successors = problem.getSuccessors(currentNode)         # Getting all successors
        for successor, path, cost in successors:
            if successor not in visited:
                totalpath = pathtillnow + [path]
                fringeList.push((successor, totalpath))

    #return False

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    startState = problem.getStartState()
    fringeList = util.PriorityQueue()
    visited = []
    fringeList.push((startState,[],0), 0)                                 # Storing new node in priority queue
    while fringeList:
        currentNode, pathtillnow, costtillnow = fringeList.pop()
        visited.append(currentNode)
        if problem.isGoalState(currentNode):
            return pathtillnow;
        successors = problem.getSuccessors(currentNode)         # Getting all successors
        for successor, path, cost in successors:
            if successor not in visited:
                totalpath = pathtillnow + [path]
                fringeList.push((successor, totalpath, cost + costtillnow), cost + costtillnow)                 # pushing successsor and cost in priority queue


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    startState = problem.getStartState()
    fringeList = util.PriorityQueue()
    visited = []
    fringeList.push((startState,[],0), 0)                                 # Storing new node in priority queue
    while fringeList:
        currentNode, pathtillnow, costtillnow = fringeList.pop()
        visited.append(currentNode)
        if problem.isGoalState(currentNode):
            return pathtillnow;
        successors = problem.getSuccessors(currentNode)         # Getting all successors
        for successor, path, cost in successors:
            if successor not in visited:
                totalpath = pathtillnow + [path]
                fringeList.push((successor, totalpath, cost + costtillnow + heuristic(successor, problem)), cost + costtillnow + heuristic(successor, problem))                 # pushing successsor and cost in priority queue


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
