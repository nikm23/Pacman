# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

STOPPENALTY = -1.0
BONUS = 1.0

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        ghostDistances = [manhattanDistance(newPos, ghostPosition.configuration.pos) for ghostPosition in
                          newGhostStates]
        closestGhost = min(ghostDistances)
        foodDistances = []
        score = 0
        for foodDistance in newFood.asList():
            foodDistances.append(manhattanDistance(newPos, foodDistance))
            if len(newFood.asList()) < len(currentGameState.getFood().asList()):
                score = BONUS                                     # when net action leads to consumption of food,
                                                                    # we give bonus score to that move.
        if closestGhost <= 1:
            score = -99999.0
        elif action == "Stop":                                    # Whenever pacman stops, we give stop penalty
            score = STOPPENALTY
        else:
            if len(foodDistances) > 0:
                minFoodDistance = min(foodDistances)              # getting minimum food distance and
                score = score + 1.0/minFoodDistance               # taking reciprocal of it to give it highest score
            else:
                score = score + 9999.0                            # this case is when we consume last food and wins the game

        return float(score)




def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def minimax(self, curDepth, targetDepth, agentIndex, numagents, gameState):

        if gameState.isWin() or gameState.isLose() or (curDepth == targetDepth):         #checking if game is won or lost or reached final depth
            return self.evaluationFunction(gameState)

        if (agentIndex == 0):                                                    #this is the max node
            scores = []
            legalActions = gameState.getLegalActions()                           #for every legal action we have a branch
            for action in legalActions:
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                scores.append(self.minimax(curDepth, targetDepth, agentIndex+1, numagents, successorGameState))
            bestScore = max(scores)                                              # for max taking maximum score
            return bestScore

        elif(agentIndex < numagents-1):                                         # these min are for all ghost agents except last
            scores = []
            legalActions = gameState.getLegalActions(agentIndex)
            for action in legalActions:
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                scores.append(
                    self.minimax(curDepth, targetDepth, agentIndex + 1, numagents, successorGameState))     #Since ghost try to kill pacman,
                                                                                                            #he will try to minimize score
            worstScore = min(scores)
            return worstScore

        elif (agentIndex == numagents - 1):                                                   #this is for last ghost agent
            scores = []
            legalActions = gameState.getLegalActions(agentIndex)
            for action in legalActions:
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                scores.append(
                    self.minimax(curDepth + 1, targetDepth, 0, numagents, successorGameState))      #now next agent is of pacman,
                                                                                                    # thus 0 is passed and depth increased
            worstScore = min(scores)
            return worstScore

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        numagents = gameState.getNumAgents()
        agentIndex = 0
        scores = []
        legalActions = gameState.getLegalActions()
        for action in legalActions:                                                            #Pacman needs to choose action which is best
            successorGameState = gameState.generateSuccessor(0, action)
            scores.append(self.minimax(0, self.depth, agentIndex + 1, numagents, successorGameState))
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]   #Picks the best score
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
        return legalActions[chosenIndex]          #Picks action related to best score

class AlphaBetaAgent(MultiAgentSearchAgent):
    MAX, MIN = 10000, -10000
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def alphaBeta(self, curDepth, targetDepth, agentIndex, numagents, gameState, alpha, beta):

        if gameState.isWin() or gameState.isLose() or (curDepth == targetDepth):
            return self.evaluationFunction(gameState)

        if (agentIndex == 0):
            best = self.MIN
            legalActions = gameState.getLegalActions()
            for action in legalActions:
                successorGameState = gameState.generateSuccessor(agentIndex, action)             #Getting game state after pacman's legal actions
                score = self.alphaBeta(curDepth, targetDepth, agentIndex + 1, numagents, successorGameState, alpha, beta)
                best = max(best, score)                     #getting the best score for max node
                alpha = max(alpha, best)
                if beta < alpha:
                    break                                   #this is the pruning step
            return best

        elif(agentIndex < numagents-1):            #For all ghost agents except the last
            best = self.MAX
            legalActions = gameState.getLegalActions(agentIndex)
            for action in legalActions:
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                score = self.alphaBeta(curDepth, targetDepth, agentIndex + 1, numagents, successorGameState, alpha, beta) #Agent increased by one for next ghost
                best = min(best, score)
                beta = min(beta, best)
                # Alpha Beta Pruning
                if beta < alpha:
                    break
            return best

        elif (agentIndex == numagents - 1):                  #Last ghost agent
            best = self.MAX
            legalActions = gameState.getLegalActions(agentIndex)
            for action in legalActions:
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                score = self.alphaBeta(curDepth + 1, targetDepth, 0, numagents, successorGameState, alpha, beta)
                                                        #Passing 0 as agent Index as next is max node and increasing depth by 1
                best = min(best, score)
                beta = min(beta, best)
                # Alpha Beta Pruning
                if beta < alpha:
                    break
            return best

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numagents = gameState.getNumAgents()
        agentIndex = 0
        scores = []
        alpha = self.MIN
        beta = self.MAX
        legalActions = gameState.getLegalActions()
        for action in legalActions:                            #Getting the all legal actions for pacman
            best = self.MIN
            successorGameState = gameState.generateSuccessor(0, action)
            score = self.alphaBeta(0, self.depth, agentIndex + 1, numagents, successorGameState, alpha, beta)
            scores.append(score)
            best = max(best, score)
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        bestScore = max(scores)                           #Getting the best score
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
        return legalActions[chosenIndex]          #returning the best action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def expectimax(self, curDepth, targetDepth, agentIndex, numagents, gameState):

        if gameState.isWin() or gameState.isLose() or (curDepth == targetDepth):
            return self.evaluationFunction(gameState)

        if (agentIndex == 0):
            scores = []
            legalActions = gameState.getLegalActions()
            for action in legalActions:
                successorGameState = gameState.generateSuccessor(agentIndex, action)           #Getting all game state after pacman's legal action
                scores.append(self.expectimax(curDepth, targetDepth, agentIndex+1, numagents, successorGameState))
            bestScore = max(scores)                                   #As pacman always try to maximize the score, it will choose max score.
            return bestScore

        elif(agentIndex < numagents-1):                                   # done for all ghost agents except the last one
            scores = []
            legalActions = gameState.getLegalActions(agentIndex)
            for action in legalActions:
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                scores.append(
                    self.expectimax(curDepth, targetDepth, agentIndex + 1, numagents, successorGameState))
            probScore = 1.0*sum(scores)/len(scores)                                      #As ghost may not choose optimal path, we pick avg score
            return probScore

        elif (agentIndex == numagents - 1):                  #Done for last ghost agent
            scores = []
            legalActions = gameState.getLegalActions(agentIndex)                            #Getting all legal actions
            for action in legalActions:
                successorGameState = gameState.generateSuccessor(agentIndex, action)        #Getting game states for all legal actions
                scores.append(
                    self.expectimax(curDepth + 1, targetDepth, 0, numagents, successorGameState))    #As next will be pacman, agentIndex is passed as 1
            probScore = 1.0*sum(scores)/len(scores)
            return probScore



    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        numagents = gameState.getNumAgents()
        agentIndex = 0
        scores = []
        legalActions = gameState.getLegalActions()
        for action in legalActions:
            successorGameState = gameState.generateSuccessor(0, action)
            scores.append(self.expectimax(0, self.depth, agentIndex + 1, numagents, successorGameState))
        bestScore = max(scores)                                                   #Getting max score of the all legal actions
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
        return legalActions[chosenIndex]           #Picking the best legal action

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

