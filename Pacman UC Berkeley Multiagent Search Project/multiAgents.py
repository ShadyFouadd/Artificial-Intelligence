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
        min = 999999999
        dist = 0
        currFood = currentGameState.getFood()
        foodListIndex = currFood.asList()
        for k in foodListIndex:
            dist = (manhattanDistance(k,newPos))
            if dist < min:
                min = dist
        score = -min
        for ghost in newGhostStates:
            if ghost.scaredTimer == 0 and ghost.getPosition() == newPos:
                return -999999999

        if action == 'Stop':
            return -999999999
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

        ghostsCount = gameState.getNumAgents() - 1

        # A function to check if the state is terminal for max player
        def checkTerminalMax(gameState, depth):
            if gameState.isWin() or gameState.isLose() or depth + 1 == self.depth:
                return True
            return False

        def checkTerminalMin(gameState, depth):
            if gameState.isWin() or gameState.isLose():
                return True
            return False

        def maxPlayer(gameState, depth):
            """Checking if the state is terminal already"""
            currentDepth = depth + 1
            if checkTerminalMax(gameState, depth):  # Terminal Test
                return self.evaluationFunction(gameState)
            maxVal = -969696
            possibleActions = gameState.getLegalActions(0)
            for action in possibleActions:
                successor = gameState.generateSuccessor(0, action)
                maxVal = max(maxVal, minPlayer(successor, currentDepth, 1))
            return maxVal

        def minPlayer(gameState, depth, index):
            minVal = 969696
            if checkTerminalMin(gameState, depth):  # Terminal Test
                return self.evaluationFunction(gameState)
            possibleActions = gameState.getLegalActions(index)
            for action in possibleActions:
                successor = gameState.generateSuccessor(index, action)
                if index == (gameState.getNumAgents() - 1):
                    minVal = min(minVal, maxPlayer(successor, depth))
                else:
                    minVal = min(minVal, minPlayer(successor, depth, index + 1))
            return minVal

        # Initiation
        possibleActions = gameState.getLegalActions(0)
        currentScore = -969696
        thisAction = None
        for action in possibleActions:
            nextState = gameState.generateSuccessor(0, action)
            newScore = minPlayer(nextState, 0, 1)
            if newScore > currentScore:
                thisAction = action
                currentScore = newScore
        return thisAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        inf = float('inf')
        initalpha = -inf
        initbeta = inf
        action, score = self.alpha_beta(0, 0, gameState, initalpha, initbeta)  # Get the action and score for pacman (max)
        return action


    def alpha_beta(self, currdepth, agenti, gameState, alpha, beta):

            # increase current depth if all agents have finished playing their turn in a move
            if agenti >= gameState.getNumAgents():
                agenti = 0
                currdepth += 1
            if currdepth == self.depth:
                return None, self.evaluationFunction(gameState)
            best_score, best_action = None, None

            for action in gameState.getLegalActions(agenti):
                nextstate = gameState.generateSuccessor(agenti, action)
                _, score = self.alpha_beta(currdepth, agenti + 1, nextstate, alpha, beta)
                # pacman turn
                if agenti == 0:
                    if best_score is None or score > best_score:
                        best_action = action
                        best_score = score
                    alpha = max(score, alpha)
                # ghost turn
                else:
                    if best_score is None or score < best_score:
                        best_action = action
                        best_score = score
                    beta = min(score, beta)
                # Prune the tree if alpha is greater than beta
                if alpha > beta:
                    break
            if best_score is None:
                return None, self.evaluationFunction(gameState)
            return best_action, best_score
            util.raiseNotDefined()

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
        def expectiScore(gameState, ghost_turn, depth_curr):
            ghostactions = gameState.getLegalActions(ghost_turn)
            bestscore = 0
            for action in ghostactions:
                next_state = gameState.generateSuccessor(ghost_turn, action)
                score,_ = expectimax(next_state, ghost_turn+1, depth_curr)
                bestscore+=score
            return bestscore/len(ghostactions)

        def maxScore(gameState, depth_curr):
            bestscore, bestaction=None, None
            for action in gameState.getLegalActions(0):  # max player is always pacman whose turn is zero
                next_state = gameState.generateSuccessor(0, action)
                score,_ = expectimax(next_state, 1, depth_curr)
                if bestscore is None or score > bestscore:
                    bestscore,bestaction = score,action
            return bestscore, bestaction

        #depending on whose move and depth, either maximise or get average score
        def expectimax(gameState,agent_turn,depth_curr):
            if agent_turn>=gameState.getNumAgents():
                agent_turn=0
                depth_curr+=1
            if gameState.isWin() or gameState.isLose() or depth_curr == self.depth:
                return  self.evaluationFunction(gameState),None
            if agent_turn==0:
                return maxScore(gameState,depth_curr)
            else:
                return expectiScore(gameState,agent_turn,depth_curr),None
        score, action = expectimax(gameState, 0, 0)
        return action


        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    currPos = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    # Distance from the ghosts to current pacman
    ghostsDist = []
    for ghost in ghostStates:
        ghostsDist.append(manhattanDistance(ghost.getPosition(), currPos))

    numCaps = len(currentGameState.getCapsules())

    # Now we calculate the food distances
    foodList = currentGameState.getFood().asList()
    foodDist = []
    for food in foodList:
        foodDist.append(manhattanDistance(currPos, food))

    # Now we calculate the scores
    score = currentGameState.getScore()
    totalScaredTimes = sum(scaredTimes)
    totalGhostDist = sum(ghostsDist)
    inverseFoodDist = 0

    if totalScaredTimes > 1:
        score +=  5*totalScaredTimes + (-2 * numCaps) + (-1 * totalGhostDist)
    else:
        score += 2*totalGhostDist + 1*numCaps

    # The closer the food the better the score
    if len(foodDist) > 0:
        inverseFoodDist = 1.5 / min(foodDist)
    score += inverseFoodDist

    # Lastly, pacman is rewarded for eating the food
    emptyList = currentGameState.getFood().asList(False)
    emptyFood = len(emptyList)
    score += 0.5*emptyFood

    return score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
