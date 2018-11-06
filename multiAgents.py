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

        food = currentGameState.getFood()
        currentPos = list(successorGameState.getPacmanPosition())
        distance = float("-Inf")

        foodList = food.asList()

        if action == 'Stop':
            return float("-Inf")

        for state in newGhostStates:
            if state.getPosition() == tuple(currentPos) and (state.scaredTimer == 0):
                return float("-Inf")

        for x in foodList:
            tempDistance = -1 * (manhattanDistance(currentPos, x))
            if (tempDistance > distance):
                distance = tempDistance

        return distance



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

    def __init__(self, index, evalFn = 'betterEvaluationFunction', depth = '2'):
        self.index = index # Pacman is always agent index 0
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
        """
        "*** YOUR CODE HERE ***"
        def minValue(gameState, depth, agentcounter):
            minimum = ["", float("inf")]
            ghostActions = gameState.getLegalActions(agentcounter)

            if not ghostActions:
                return self.evaluationFunction(gameState)

            for action in ghostActions:
                currState = gameState.generateSuccessor(agentcounter, action)
                current = minOrMax(currState, depth, agentcounter + 1)
                if type(current) is not list:
                    newVal = current
                else:
                    newVal = current[1]
                if newVal < minimum[1]:
                    minimum = [action, newVal]
            return minimum

        def maxValue(gameState, depth, agentcounter):
            maximum = ["", -float("inf")]
            actions = gameState.getLegalActions(agentcounter)

            if not actions:
                return self.evaluationFunction(gameState)

            for action in actions:
                currState = gameState.generateSuccessor(agentcounter, action)
                current = minOrMax(currState, depth, agentcounter + 1)
                if type(current) is not list:
                    newVal = current
                else:
                    newVal = current[1]
                if newVal > maximum[1]:
                    maximum = [action, newVal]
            return maximum


        def minOrMax(gameState, depth, agentcounter):
            if agentcounter >= gameState.getNumAgents():
                depth += 1
                agentcounter = 0

            if (depth == self.depth or gameState.isWin() or gameState.isLose()):
                return self.evaluationFunction(gameState)
            elif (agentcounter == 0):
                return maxValue(gameState, depth, agentcounter)
            else:
                return minValue(gameState, depth, agentcounter)

        actionsList = minOrMax(gameState, 0, 0)
        return actionsList[0]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def minValue(gameState, depth, agentcounter, a, b):
            minimum = ["", float("inf")]
            ghostActions = gameState.getLegalActions(agentcounter)

            if not ghostActions:
                return self.evaluationFunction(gameState)

            for action in ghostActions:
                currState = gameState.generateSuccessor(agentcounter, action)
                current = minOrMax(currState, depth, agentcounter + 1, a, b)

                if type(current) is not list:
                    newVal = current
                else:
                    newVal = current[1]

                if newVal < minimum[1]:
                    minimum = [action, newVal]
                if newVal < a:
                    return [action, newVal]
                b = min(b, newVal)
            return minimum

        def maxValue(gameState, depth, agentcounter, a, b):
            maximum = ["", -float("inf")]
            actions = gameState.getLegalActions(agentcounter)

            if not actions:
                return self.evaluationFunction(gameState)

            for action in actions:
                currState = gameState.generateSuccessor(agentcounter, action)
                current = minOrMax(currState, depth, agentcounter + 1, a, b)

                if type(current) is not list:
                    newVal = current
                else:
                    newVal = current[1]

                # real logic
                if newVal > maximum[1]:
                    maximum = [action, newVal]
                if newVal > b:
                    return [action, newVal]
                a = max(a, newVal)
            return maximum

        def minOrMax(gameState, depth, agentcounter, a, b):
            if agentcounter >= gameState.getNumAgents():
                depth += 1
                agentcounter = 0

            if (depth == self.depth or gameState.isWin() or gameState.isLose()):
                return self.evaluationFunction(gameState)
            elif (agentcounter == 0):
                return maxValue(gameState, depth, agentcounter, a, b)
            else:
                return minValue(gameState, depth, agentcounter, a, b)

        actionsList = minOrMax(gameState, 0, 0, -float("inf"), float("inf"))
        return actionsList[0]

class ExpectimaxAgent(MultiAgentSearchAgent):

    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        # print(self.index)
        root_value = self.value(gameState,0,self.index)
        action = root_value[1]
        return action

    def value(self,gameState,CurrentDepth,agentIndex):

        if agentIndex == gameState.getNumAgents():
            CurrentDepth = CurrentDepth + 1
            agentIndex = agentIndex = 0

        legal_action = []
        legal_action = gameState.getLegalActions(agentIndex)

        if len(legal_action) == 0:
            eval_value =  self.evaluationFunction(gameState)
            return [eval_value]

        if CurrentDepth == self.depth:
            eval_value =  self.evaluationFunction(gameState)
            return [eval_value]

        if agentIndex == 0:
            return self.max_value(gameState,CurrentDepth,agentIndex)
        else:
            return self.min_value(gameState,CurrentDepth,agentIndex)

    def max_value(self,gameState,CurrentDepth,agentIndex):

        node_value = [-float("inf")]

        action_possible = []
        action_possible = gameState.getLegalActions(agentIndex)

        for action in action_possible:
            successor_state = gameState.generateSuccessor(agentIndex, action)
            successor_evalvalue = self.value(successor_state, CurrentDepth, agentIndex + 1)

            successor_evalvalue = successor_evalvalue[0]

            if (successor_evalvalue >= node_value[0]):
                node_value = [successor_evalvalue,action]

        return node_value

    def min_value(self,gameState,CurrentDepth,agentIndex):

        node_value = [float("inf")]

        action_list = []
        action_list = gameState.getLegalActions(agentIndex)

        for action in action_list:
            successor_state = gameState.generateSuccessor(agentIndex, action)
            successor_evalvalue = self.value(successor_state, CurrentDepth, agentIndex + 1)

            successor_evalvalue = successor_evalvalue[0]

            if (successor_evalvalue <= node_value[0]):
                node_value = [successor_evalvalue,action]

        return node_value



def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """

    position = list(currentGameState.getPacmanPosition())
    foodPos = currentGameState.getFood().asList()
    foodList = []

    for food in foodPos:
        pacmanDist = manhattanDistance(position, food)
        foodList.append(pacmanDist)

    if not foodList:
        foodList.append(0)

    nearestPelletDist = min(foodList)
    return currentGameState.getScore() + (-1) * nearestPelletDist

# Abbreviation
better = betterEvaluationFunction
