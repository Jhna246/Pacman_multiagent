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
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # mdistance returns abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

        v = successorGameState.getScore()
        closest_food = 0

        # print(newGhostStates[0].getPosition())
        # print(newPos, 'myposition')

        # I guess this will calculate pacman's position from ghost
        closest_ghost = manhattanDistance(newPos, newGhostStates[0].getPosition())

        food = newFood.asList()
        # for every food in the map, find how far each food is from pacman's position
        distance_of_food = [manhattanDistance(newPos, i) for i in food]

        # find the min distance because thats where pacman should go since it's closest.
        if len(food) > 1:
            closest_food = min(distance_of_food)
        elif len(food) == 1:
            closest_food = distance_of_food[0]
            # print(closest_food, 'lossodijasefawese')

        # if a ghost is very close to pacman, get away from it or I will lose. Just make val decrease by a big number
        if closest_ghost <= 3:
            v -= 100

        if len(distance_of_food) > 0:
            v += 1 / closest_food    # makes some back and forth movements but a lot faster than just closest_food
            # increasing the 1 above does not do anything to make the program better. I guess I just needed to divide.

            # val += closest_food   # doesn't work pacman doesn't moves back and forth infinitely
            # val += closest_food * 10  # It works but it makes back and forth movements a lot. Needs ghost to be near it to stop
        # print(val)

        return v

        # There are ways to optimize this code. There are times where pacman makes the same move over and over again
        # and needs the help of ghost to stop the repetitive movement. I think this is because when pacman finds closest
        # food them moves towards that position, when it gets to the new position it finds a different closest food pos.
        # also not optimized when there are two ghosts

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

        action = "" # action that I want pacman to take. Take no action as of yet
        temp_value = -1000000000 # set temp value to a very small number so we 100% take in next v

        for i in gameState.getLegalActions(0):
            suc = gameState.generateSuccessor(0, i)
            v = self.value(suc, 1)
            # print(v)
            if v > temp_value: # if the new value is bigger than the one before
                action = i
                temp_value = v
        # print(action)
        return action


    # minimax implementation from your slide
    #
    # def value(gameState):
    #     agent = gameState.getNumAgents()
    #     if agent == 0:
    #         return max_value(gameState)
    #     elif agent > 0:
    #         return min_value(gameState, agent)

    # def max_value(gameState):
    #     v = float('-inf')
    #     for i in gameState.getLegalActions(0):
    #         v = max(v, value(gameState.generateSuccessor(0, i)))
    #     return v
    #
    # def min_value(gameState, agent):
    #     v = float('-inf')
    #     for i in gameState.getLegalActions(agent):
    #         v = min(v, value(gameState.generateSuccessor(agent, i)))
    #     return v

    def value(self, gameState, depth):

        # according to the message in command prompt:
        # there is a tree with 7 depth it tells us where agent = 2
        # max is located at depth = 0, 2, 4
        # we can calculate max by finding the depth % agent == 0
        # a - max
        # b - min
        # c - max
        # d - min
        # e - max
        # f - min

        agent = depth % gameState.getNumAgents()
        agents = gameState.getNumAgents()
        # print(depth, gameState.getNumAgents(), 'depth, agent')
        #
        # print(depth, 'depth')
        # print(self.depth, 'self.depth')

        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        elif depth == self.depth * agents: # self.depth * agents is max depth of tree
            # print(depth, self.depth * agents)
            return self.evaluationFunction(gameState)

        elif depth % agents == 0:   # explanation above
            # print(max_value(gameState, depth, agent))
            return self.max_value(gameState, depth)
        else:
            # print(min_value(gameState, depth, agent))
            return self.min_value(gameState, depth, agent)

    def max_value(self, gameState, depth):
        v = float("-inf")
        action = gameState.getLegalActions(0)
        for i in action:
            suc = gameState.generateSuccessor(0, i)
            v = max(v, self.value(suc, depth + 1))
            # print(self.value(suc, depth + 1))
            # print(v, 'max value')
        return v

    def min_value(self, gameState, depth, agent):
        v = float("inf")
        action = gameState.getLegalActions(agent)
        for i in action:
            suc = gameState.generateSuccessor(agent, i)
            v = min(v, self.value(suc, depth + 1))
        # print(v, 'min value')
        return v


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        action = ""  # action that I want pacman to take. Take no action as of yet
        temp_value = -1000000000  # set temp value to a very small number so we 100% take in next v
        alpha = float("-inf")

        for i in gameState.getLegalActions(0):
            suc = gameState.generateSuccessor(0, i)
            v = self.value(suc, 1, alpha, float("inf"))
            # print(v)
            if v > temp_value:  # if the new value is bigger than the one before
                action = i
                temp_value = v
            alpha = max(alpha, temp_value)
        # print(action)
        return action

        # from slide
        #         #def value(gameState, depth, alpha, beta):
        #         #     if depth == self.depth:
        #         #         return self.evaluationFunction(gameState)
        #         #
        #         # def max_value(gameState, alpha, beta):
        #         #     v = float("-inf")
        #         #     action = gameState.getLegalActions(0)
        #         #     for i in action:
        #         #         v = max(v, value(gameState.generateSuccessor(agent, i), alpha, beta))
        #         #         if v > beta:
        #         #             return v
        #         #         alpha = max(alpha, v)
        #         #     return v
        #         #
        #         # def min_value(gameState, alpha, beta):
        #         #     v = float("inf")
        #         #     action = gameState.getLegalActions(0)
        #         #     for i in action:
        #         #         v = min(v, value(gameState.generateSuccessor(agent, i), alpha, beta))
        #         #         if v > beta:
        #         #             return v
        #         #         alpha = min(alpha, v)
        #         #     return v

    def value(self, gameState, depth, alpha, beta):

        agent = depth % gameState.getNumAgents()
        agents = gameState.getNumAgents()
        # print(depth, gameState.getNumAgents(), 'depth, agent')
        #
        # print(depth, 'depth')
        # print(self.depth, 'self.depth')

        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        elif depth == self.depth * agents:  # self.depth * agents is max depth of tree
            # print(depth, self.depth * agents)
            return self.evaluationFunction(gameState)

        elif depth % agents == 0:
            # print(max_value(gameState, depth, agent)) # only prints inf for some reason
            return self.max_value(gameState, depth, alpha, beta)
        else:
            # print(min_value(gameState, depth, agent))
            return self.min_value(gameState, depth, agent, alpha, beta)


    def max_value(self, gameState, depth, alpha, beta):
        v = float("-inf")
        action = gameState.getLegalActions(0)
        for i in action:
            suc = gameState.generateSuccessor(0, i)
            v = max(v, self.value(suc, depth + 1, alpha, beta))
            if v > beta:
                # print(v, beta, 'v, beta')
                return v
            alpha = max(alpha, v)
        return v
            # print(self.value(suc, depth + 1))
            # print(v, 'max value')

    def min_value(self, gameState, depth, agent, alpha, beta):
        v = float("inf")
        action = gameState.getLegalActions(agent)
        for i in action:
            suc = gameState.generateSuccessor(agent, i)
            v = min(v, self.value(suc, depth + 1, alpha, beta))
            if v < alpha:
                # print(v, alpha, 'v alpha')
                return v
            beta = min(beta, v)
        return v
        # print(v, 'min value')


        # util.raiseNotDefined()

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
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

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
