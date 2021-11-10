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
import math

from game import Agent

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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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
        # newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]


        "*** YOUR CODE HERE ***"
        currentFood = currentGameState.getFood()
        score = successorGameState.getScore()
        updatedGhostPositions = successorGameState.getGhostPositions()
        currentFoodList = currentFood.asList()
        newFoodList = newFood.asList()
        nearestFood = math.inf
        nearestGhost = math.inf
        foodScore = 0

        # Award the Pacman for moving to a food position
        if newPos in currentFoodList:
            foodScore = 15.0

        # Compute the distance to the closest food
        distanceFromFood = [manhattanDistance(newPos, foodPos) for foodPos in newFoodList]
        allFood = len(newFoodList)
        if len(distanceFromFood):
            nearestFood = min(distanceFromFood)

        # Award score by:
        # * Adding Weighted inverse of nearest food: The closest to the food, the highest the score
        # * Subtracting remaining food to eat
        # * Adding the score of moving to a food position
        score += 15.0 / nearestFood - 5.0 * allFood + foodScore

        # Compute distance to the nearest ghost
        for n in updatedGhostPositions:
            distance = manhattanDistance(newPos, n)
            nearestGhost = min([nearestGhost, distance])

        # If the ghost is very close, we subtract a significant amount of points
        if nearestGhost < 5:
            score -= 100.0

        return score


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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.maxAgent(gameState, agentIndex = 0, depth = 0)[0]

    '''
        Returns the evaluation value of the given state for the given agent (Pacman - 0 or Ghost - >= 1) at the 
        specified depth
    '''
    def value(self, gameState, agentIndex, depth):
        # Max depth reached, lose state or win state 
        if depth == gameState.getNumAgents() * self.depth or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        # Evaluate for Pacman
        if agentIndex == 0:
            return self.maxAgent(gameState, agentIndex, depth)[1]
        # Evaluate for Ghost
        else:
            return self.minAgent(gameState, agentIndex, depth)[1]

    '''
        Returns the best possible action for the maximising agent
    '''
    def maxAgent(self, gameState, agentIndex, depth):
        bestAction = ("max", -math.inf)
        for action in gameState.getLegalActions(agentIndex):
            succAction = (action, self.value(gameState.generateSuccessor(agentIndex, action),
                                             (depth + 1) % gameState.getNumAgents(), depth + 1))
            bestAction = max(bestAction, succAction, key=lambda x: x[1])
        return bestAction

    '''
        Returns the best possible action for the minimising agent
    '''
    def minAgent(self, gameState, agentIndex, depth):
        bestAction = ("min", math.inf)
        for action in gameState.getLegalActions(agentIndex):
            succAction = (action, self.value(gameState.generateSuccessor(agentIndex, action),
                                             (depth + 1) % gameState.getNumAgents(), depth + 1))
            bestAction = min(bestAction, succAction, key=lambda x: x[1])
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.maxAgent(gameState, 0, 0, - math.inf, math.inf)[0]

    '''
        Returns the evaluation value of the given state for the given agent (Pacman - 0 or Ghost - >= 1) at the 
        specified depth
    '''
    def alphabeta(self, gameState, agentIndex, depth, alpha, beta):
        # Max depth reached, lose state or win state 
        if depth == gameState.getNumAgents() * self.depth or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        # Evaluate for Pacman
        if agentIndex == 0:
            return self.maxAgent(gameState, agentIndex, depth, alpha, beta)[1]
        # Evaluate for Ghost
        else:
            return self.minAgent(gameState, agentIndex, depth, alpha, beta)[1]

    '''
        Returns the best possible action for the maximising agent
    '''
    def maxAgent(self, gameState, agentIndex, depth, alpha, beta):
        bestAction = ("max", -math.inf)
        for action in gameState.getLegalActions(agentIndex):
            succAction = (action, self.alphabeta(gameState.generateSuccessor(agentIndex, action),
                                                 (depth + 1) % gameState.getNumAgents(), depth + 1, alpha, beta))
            bestAction = max(bestAction, succAction, key=lambda x: x[1])

            if bestAction[1] > beta:
                return bestAction
            else:
                alpha = max(alpha, bestAction[1])

        return bestAction

    '''
        Returns the best possible action for the minimising agent
    '''
    def minAgent(self, gameState, agentIndex, depth, alpha, beta):
        bestAction = ("min", math.inf)
        for action in gameState.getLegalActions(agentIndex):
            succAction = (action, self.alphabeta(gameState.generateSuccessor(agentIndex, action),
                                                 (depth + 1) % gameState.getNumAgents(), depth + 1, alpha, beta))
            bestAction = min(bestAction, succAction, key=lambda x: x[1])

            if bestAction[1] < alpha:
                return bestAction
            else:
                beta = min(beta, bestAction[1])

        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
          The expectimax function returns a tuple of (actions,
        """
        "*** YOUR CODE HERE ***"
        maxDepth = self.depth * gameState.getNumAgents()
        return self.expectimax(gameState, "expect", maxDepth, 0)[0]


    def expectimax(self, gameState, action, depth, agentIndex):
        if depth is 0 or gameState.isLose() or gameState.isWin():
            return (action, self.evaluationFunction(gameState))

        if agentIndex is 0:
            return self.maxvalue(gameState, action, depth, agentIndex)
        else:
            return self.expvalue(gameState, action, depth, agentIndex)

    '''
        Returns the best possible action for the maximising agent
    '''
    def maxvalue(self, gameState, action, depth, agentIndex):
        bestAction = ("max", -(math.inf))
        for legalAction in gameState.getLegalActions(agentIndex):
            nextAgent = (agentIndex + 1) % gameState.getNumAgents()
            succAction = None
            if depth != self.depth * gameState.getNumAgents():
                succAction = action
            else:
                succAction = legalAction
            succValue = self.expectimax(gameState.generateSuccessor(agentIndex, legalAction),
                                        succAction, depth - 1, nextAgent)
            bestAction = max(bestAction, succValue, key=lambda x: x[1])
        return bestAction

    '''
        Returns the best possible action for the minimising agent (using expectimax)
    '''
    def expvalue(self, gameState, action, depth, agentIndex):
        legalActions = gameState.getLegalActions(agentIndex)
        averageScore = 0
        propability = 1.0 / len(legalActions)
        for legalAction in legalActions:
            nextAgent = (agentIndex + 1) % gameState.getNumAgents()
            bestAction = self.expectimax(gameState.generateSuccessor(agentIndex, legalAction),
                                         action, depth - 1, nextAgent)
            averageScore += bestAction[1] * propability
        return (action, averageScore)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()

    # Get minimum distance to closest food
    minFoodDist = math.inf
    for food in newFood:
        minFoodDist = min(minFoodDist, manhattanDistance(newPos, food))

    # Get the minimum distance to the closest ghost
    minGhostDist = math.inf
    for ghost in currentGameState.getGhostPositions():
        minGhostDist = min(minGhostDist, manhattanDistance(newPos, ghost))

    # Ghost is to close, return a very bad score
    if (minGhostDist < 2):
        return -float('inf')

    capsLeft = len(currentGameState.getCapsules())
    foodLeft = currentGameState.getNumFood()
    foodMultiplier = 10000
    capsMultiplier = 5000
    foodDistMultiplier = 200
    additionalFactors = 0

    # Give a high score to win state and low score to lose
    if currentGameState.isLose():
        additionalFactors -= 50000
    elif currentGameState.isWin():
        additionalFactors += 50000

    # Weighted sum of scores:
    # * Inverse of amount of food left: the less food left, the higher the score
    # * Minimum distance to ghost: the closest the ghost, the worse
    # * Inverse of minimum distance to food: the closest the food, the better
    # * Inverse of amount of capsules left: the less capsules left, the higher the score
    return 1.0/(foodLeft + 1) * foodMultiplier + minGhostDist + 1.0/(minFoodDist + 1) * foodDistMultiplier + 1.0/(capsLeft + 1) * capsMultiplier + additionalFactors

# Abbreviation
better = betterEvaluationFunction
