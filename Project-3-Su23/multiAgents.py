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
import random
import util

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        # Initialize the evaluation score
        evaluationScore = 0

        # Distance to the closest food
        foodList = newFood.asList()
        minFoodDistance = min([util.manhattanDistance(newPos, food)
                               for food in foodList]) if foodList else 0
        # Reciprocal of distance
        evaluationScore += 1.0 / (minFoodDistance + 1)

        # Distance to ghosts and their scared times
        for i, ghostState in enumerate(newGhostStates):
            ghostDistance = util.manhattanDistance(
                newPos, ghostState.getPosition())
            if newScaredTimes[i] > 0:  # Ghost is scared
                evaluationScore += 1.0 / (ghostDistance + 1)
            else:  # Ghost is not scared
                if ghostDistance <= 1:
                    evaluationScore -= 100  # Very high penalty if too close to ghost

        # Current game score
        evaluationScore += successorGameState.getScore()

        return evaluationScore


def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    def minimax(self, gameState, depth, agentIndex):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState), None

        numAgents = gameState.getNumAgents()
        legalActions = gameState.getLegalActions(agentIndex)

        if agentIndex == 0:  # Max-agent (Pacman)
            bestValue = float('-inf')
            bestAction = None
            for action in legalActions:
                successorState = gameState.generateSuccessor(
                    agentIndex, action)
                value, _ = self.minimax(
                    successorState, depth, (agentIndex + 1) % numAgents)
                if value > bestValue:
                    bestValue, bestAction = value, action
            return bestValue, bestAction

        else:  # Min-agent (Ghosts)
            bestValue = float('inf')
            bestAction = None
            for action in legalActions:
                successorState = gameState.generateSuccessor(
                    agentIndex, action)
                if (agentIndex + 1) % numAgents == 0:
                    value, _ = self.minimax(
                        successorState, depth - 1, 0)  # next depth level
                else:
                    value, _ = self.minimax(
                        successorState, depth, (agentIndex + 1) % numAgents)

                if value < bestValue:
                    bestValue, bestAction = value, action
            return bestValue, bestAction

    def getAction(self, gameState: GameState):
        # Start minimax from the root node, which is a max node
        _, action = self.minimax(gameState, self.depth, 0)
        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    def max_value(self, gameState, depth, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)

        v = float('-inf')
        for action in gameState.getLegalActions(0):
            successorState = gameState.generateSuccessor(0, action)
            v = max(v, self.value(successorState, depth, 1, alpha, beta))
            if v > beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(self, gameState, depth, agentIndex, alpha, beta):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        v = float('inf')
        for action in gameState.getLegalActions(agentIndex):
            successorState = gameState.generateSuccessor(agentIndex, action)
            if (agentIndex + 1) % gameState.getNumAgents() == 0:
                nextValue = self.value(
                    successorState, depth - 1, 0, alpha, beta)
            else:
                nextValue = self.value(
                    successorState, depth, agentIndex + 1, alpha, beta)

            v = min(v, nextValue)
            if v < alpha:
                return v
            beta = min(beta, v)
        return v

    def value(self, gameState, depth, agentIndex, alpha, beta):
        if agentIndex == 0:
            return self.max_value(gameState, depth, alpha, beta)
        else:
            return self.min_value(gameState, depth, agentIndex, alpha, beta)

    def getAction(self, gameState: GameState):
        alpha = float('-inf')
        beta = float('inf')
        bestValue = float('-inf')
        bestAction = None

        for action in gameState.getLegalActions(0):
            successorState = gameState.generateSuccessor(0, action)
            value = self.value(successorState, self.depth, 1, alpha, beta)

            if value > bestValue:
                bestValue, bestAction = value, action

            alpha = max(alpha, bestValue)

        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    def expectimax(self, gameState, depth, agentIndex):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState), None

        numAgents = gameState.getNumAgents()
        legalActions = gameState.getLegalActions(agentIndex)

        if agentIndex == 0:  # Max-agent (Pacman)
            bestValue = float('-inf')
            bestAction = None
            for action in legalActions:
                successorState = gameState.generateSuccessor(
                    agentIndex, action)
                value, _ = self.expectimax(
                    successorState, depth, (agentIndex + 1) % numAgents)
                if value > bestValue:
                    bestValue, bestAction = value, action
            return bestValue, bestAction

        else:  # Average-agent (Ghosts)
            avgValue = 0
            for action in legalActions:
                successorState = gameState.generateSuccessor(
                    agentIndex, action)
                if (agentIndex + 1) % numAgents == 0:
                    value, _ = self.expectimax(
                        successorState, depth - 1, 0)  # next depth level
                else:
                    value, _ = self.expectimax(
                        successorState, depth, (agentIndex + 1) % numAgents)

                avgValue += value

            avgValue = avgValue / len(legalActions) if legalActions else 0
            return avgValue, None

    def getAction(self, gameState: GameState):
        # Start expectimax from the root node, which is a max node
        _, action = self.expectimax(gameState, self.depth, 0)
        return action


def betterEvaluationFunction(currentGameState: GameState):
    pacmanPos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()

    # Closest food
    foodList = food.asList()
    if len(foodList) > 0:
        minFoodDist = min(manhattanDistance(pacmanPos, foodPos)
                          for foodPos in foodList)
    else:
        minFoodDist = 0

    # Ghost distances and states
    ghostDistances = [manhattanDistance(
        pacmanPos, ghost.getPosition()) for ghost in ghostStates]
    nearGhost = min(ghostDistances) if ghostDistances else 0

    # If a ghost is too close, prioritize avoiding it
    ghostFactor = -1000 if nearGhost <= 1 else 0

    # Remaining food and capsules
    remainingFood = len(foodList)
    remainingCapsules = len(capsules)

    # Final evaluation
    return (currentGameState.getScore() +
            (10.0 / (minFoodDist + 1)) +
            ghostFactor -
            (3 * remainingFood) -
            (3 * remainingCapsules))


# Abbreviation
better = betterEvaluationFunction
