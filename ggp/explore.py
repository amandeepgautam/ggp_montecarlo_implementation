import time
from math import log, sqrt
import random
from ggp.cache import FIFOCache
from copy import deepcopy

class ExploreResponse:
    """
    Store the response for the explore action. Add more heuristics if needed later.
    """

    def __init__(self, bestMove, numHE, numTSE, numSV, time):
        self.bestMove = bestMove
        self.numHeuristicEvals = numHE
        self.numTerminalStateEvaluations = numTSE
        self.numStatesVisited = numSV
        self.time = time

class Explore:
    """
    Explore the search  graph using the Monte carlo search to
    for the best possible move.
    """
    class TreeNode:
        """
        Class to encapsulate the state properties pertaining MonteCarlo traversal.
        """

        def __init__(self, state):
            self.state = state
            self.legalMoves = None
            self.unexploredLegalMoves = None
            self.children = None
            self.parent = None
            self.bestMove = None
            self.avgReward = 0
            self.allPlayed = 0
            self.whichMove = None       #Indexed by state, to store which move gives whoch child.
            self.turnTakers = None

        def addChild(self, childNode, moves):
            """
            Using this method would break caching in the tree.
            """
            if not self.children:
                self.children = set()
            if not self.whichMove:
                self.whichMove = {}
            self.children.add(childNode)
            childNode.parent = self
            self.whichMove[childNode.state] = moves      #index the move.

        def prepareNode(self, sim, gd):
            if not self.legalMoves:
                self.legalMoves = sim.computeLegalMoves(self.state)
                self.unexploredLegalMoves = deepcopy(self.legalMoves)
            if not self.turnTakers:
                self.turnTakers = gd.turnTakers(self.legalMoves)
            if not self.children:
                self.children = set()

    class Tree:
        """
        An encapsulation for the tree structure and a cache for faster lookup
        in the tree.
        """

        def __init__(self, state):
            self.root = Explore.TreeNode(state)        #TreeNode
            self.cache = {state: self.root}

        def addChildNode(self, parentState, childState, moves):
            childNode = Explore.TreeNode(childState)
            parent = self.cache[parentState]
            parent.addChild(childNode, moves)
            self.cache[childState] = childNode

        def findStateNode(self, state):
            return self.cache.get(state)

    class PlayStatistics:
        """
        Statistics collected during play.
        """
        def __init__(self):
            self.numRandomMoves = 0
            self.totalMoves = 0

        def __str__(self):
            return 'Random moves played: ' + str(self.numRandomMoves) + \
                    'Total moves played: ' + str(self.totalMoves)

    class ExploreStatistics:
        """
        Statistics for Simulation.
        """

        def __init__(self):
            self.numTerminalStateEvals = 0
            self.numStatesVisited = 0
            self.numPathsTraversed = 0
            self.numPathsToTerminalNode = 0

        def __str__(self):
            return 'States Visited: ' + str(self.numStatesVisited) + \
                    'Terminal evaluations: ' + str(self.numTerminalStateEvals) + \
                    'Total paths traversed: ' + str(self.numPathsTraversed) + \
                    'Total paths to terminal node: ' + str(self.numPathsToTerminalNode)

    def __init__(self, gd, sim, role, shouldIStop, reportAnswer):
        self.gd = gd
        self.sim = sim
        self.role = role
        self.shouldIStop = shouldIStop
        self.reportAnswer = reportAnswer
        self.tree = None
        self.playStats = Explore.PlayStatistics()
        self.exploreStats = Explore.ExploreStatistics()

    def printStatistics(self):
        print('Explore Statistics: ')
        print(self.exploreStats)
        print('')
        print('Play Statistics: ')
        print(self.playStats)

    def selectMove(self, state):
        """
        Select the move best for the player. If not present, just play 
        a random move. Count this heuristc later.
        """
        node = self.tree.findStateNode(state)
        move = None
        self.playStats.totalMoves += 1
        if node:
            if len(node.turnTakers) == 0:
                #This node must have executed a forced move and hence will only 
                #have one children.
                move = (node.whichMove.itervalues().next())[self.role]
            elif len(node.turnTakers) == 1:
                if self.role == node.turnTakers[0]:      #maximize your chances.
                    bestReward = float("-inf")
                    for child in node.children:
                        if child.avgReward > bestReward:
                            move = (node.whichMove[child.state])[node.turnTakers[0]]
                            bestReward = child.avgReward
                else:
                    move = list(node.legalMoves[node.turnTakers[0]]).pop()
            else:
                raise ValueError("Player is not designed to handle this use case")
        else:
            self.playStats.numRandomMoves += 1
            legalMoves = self.sim.computeLegalMoves(state)
            random.shuffle(legalMoves[self.role])
            move = legalMoves[self.role].pop()

        #Update when you have heuristics.
        exploreResponse = ExploreResponse(move, 0, 0, 0, 0)
        self.reportAnswer(exploreResponse)

    def explore(self, state, heuristic=None, maxDepth=-1, maxOppMoves=-1):
        """
        Top level explore method. Intended for public use
        """
        self.heuristic = heuristic
        self.maxOppMoves = maxOppMoves

        if not self.tree:
            self.numStatesVisited = 0
            self.tree = self.Tree(state)

        if not heuristic:
            #Without heuristic intermidiate board evaluation is not possible.
            self.topLevelExplore(state, -1)
        else:
            depth = 1
            while True:
                if self.topLevelExplore(state, maxDepth):
                    break
                depth += 1
                if depth == maxDepth:
                    break

    def topLevelExplore(self, state, depth):
        #statistics
        self.exploreStats.numPathsTraversed += 1

        node = self.ucbSelection(state)
        node = self.mcExpansionAndSimulation(node.state)
        self.evaluationAndBackPropagation(node.state, None)

    def ucbSelection(self, state):
        """
        Implements the selection procedure as discussed in Monte Carlo 
        Search methods.
        """
        node = self.tree.findStateNode(state)
        if not node:
            raise ValueError("A node for the state does not exist.")

        if self.sim.isTerminal(state):
            return node

        node.prepareNode(self.sim, self.gd)

        turnTakers = node.turnTakers

        if len(turnTakers) == 0:
            #There will always be 1 child in this case as no one had a choice 
            #to move. Select that if explored, else return
            if len(node.children) == 0:
                return node
            else:
                return next(iter(node.children))

        if len(turnTakers) == 1:
            #explore until all statistics are not available.
            if len(node.unexploredLegalMoves[turnTakers[0]]) != 0:
                return node
            return self.turnTakingUCBSelection(state, turnTakers[0])
        else:
            raise ValueError("The Player cannot handle this use case")

    def turnTakingUCBSelection(self, state, turnTaker):
        """
        Implements playing logic for turntaking games.
        """
        #try to exploit better paths in the trees.
        bestChild = None
        node = self.tree.findStateNode(state)
        if self.sim.isTerminal(state):
            return node

        if self.role == turnTaker:
            maxReward = float("-inf")
            for child in node.children:
                reward = (child.avgReward) + \
                        100 * sqrt(2*log(node.allPlayed)/child.allPlayed)
                if reward > maxReward:
                    bestChild = child
                    maxReward = reward
        else:       #all opponents
            minReward = float("inf")
            for child in node.children:
                reward = (child.avgReward) -\
                        100 * sqrt(2*log(node.allPlayed)/child.allPlayed)
                if reward < minReward:
                    bestChild = child
                    minReward = reward

        return self.ucbSelection(bestChild.state)

    def findNextUnexploredState(self, state):
        """
        Find the states that have not been explored from this node yet. This function
        does not expect the state to be a terminal state. It also removes the move from
        the set of unexplored moves.
        """
        node = self.tree.findStateNode(state)

        #turnTakers = self.gd.turnTakers(node.legalMoves)
        turnTakers = node.turnTakers
        if len(turnTakers) == 0:
            #its a forced move for everyone.
            moves = [list(x).pop() for x in node.legalMoves]
            nextState = self.sim.computeNextState(state, moves)
            return nextState, moves

        if len(turnTakers) == 1:
            #turn taking game.
            #if all moves have been explored, this function should not have been called.
            if len(node.unexploredLegalMoves[turnTakers[0]]) == 0:
                return None, None

            random.shuffle(node.unexploredLegalMoves[turnTakers[0]])
            move = node.unexploredLegalMoves[turnTakers[0]].pop()
            moves = [list(x).pop() for x in node.legalMoves]
            moves[turnTakers[0]] = move

            nextState = self.sim.computeNextState(state, moves)
            return nextState, moves
        else:
            raise ValueError("The player cannot handle this use case.")

    def mcExpansionAndSimulation(self, state):
        """
        Method implements Monte Carlo expansion and simulation steps.
        """
        node = self.tree.findStateNode(state)
        node.prepareNode(self.sim, self.gd)
        # randomly select a node
        # Can implement light heuristics as well to choose nodes.
        if self.sim.isTerminal(state):
            return node

        nextState, moves = self.findNextUnexploredState(state)
        if nextState:
            self.tree.addChildNode(state, nextState, moves)
            self.exploreStats.numStatesVisited += 1
            return self.mcExpansionAndSimulation(nextState)
        else:
            #The state had no unexplored move. Hence selection proceduce should 
            #be executed followed by expansion and simulation
            newNode = self.ucbSelection(state)
            return self.mcExpansionAndSimulation(newNode.state)
            #raise ValueError("Call order of functions should be ucbSelecction->mcSimulationAndExpansion")

    def evaluationAndBackPropagation(self, state, playoutReward):
        node = self.tree.findStateNode(state)
        totalReward = node.avgReward * node.allPlayed
        node.allPlayed += 1
        if self.sim.isTerminal(state):
            #there is no point of dwelling over turntakers in terminal state.
            #No one can play in terminal state. its terminal.
            if not playoutReward:
                if node.allPlayed == 1:
                    self.exploreStats.numTerminalStateEvals += 1
                self.exploreStats.numPathsToTerminalNode += 1
                goals = self.sim.computeGoals(state)
                playoutReward = goals[self.role]
                node.avgReward = (totalReward + playoutReward)/node.allPlayed
            else:
                raise ValueError("Reward should be undefined only in case of terminal state.")
        else:
            node.avgReward = (totalReward + playoutReward)/node.allPlayed

        if node.parent:
            self.evaluationAndBackPropagation(node.parent.state, playoutReward)
