import time
import os
from math import log, sqrt
import random
from ggp.cache import FIFOCache
from copy import deepcopy

debug = 0

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
            self.parents = None
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
            self.whichMove[childNode.state] = moves      #index the move.

        def addParent(self, parentNode):
            if not self.parents:
                self.parents = set()
            self.parents.add(parentNode)

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

        def __init__(self, state, sim, gd):
            self.root = Explore.TreeNode(state)        #TreeNode
            self.root.prepareNode(sim, gd)
            self.cache = {state: self.root}

        def addChildNode(self, parentState, childState, moves):
            #see if a node already exists and if it does, just connect the 
            #parent to child. 
            childNode = self.findStateNode(childState)
            if not childNode:       #if the child does not exist, create new child
                childNode = Explore.TreeNode(childState)
            parentNode = self.cache[parentState]
            parentNode.addChild(childNode, moves)
            childNode.addParent(parentNode)
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
            return 'Random moves played: ' + str(self.numRandomMoves) + os.linesep + \
                    'Total moves played: ' + str(self.totalMoves) + os.linesep

    class ExploreStatistics:
        """
        Statistics for Simulation.
        """

        def __init__(self):
            self.numTerminalStateEvals = 0
            self.numStatesVisited = 0
            self.numPathsTraversed = 0      #without a heuristic, it would be same as numPathsToTerminalNode
            self.numPathsToTerminalNode = 0

        def __str__(self):
            return 'States Visited: ' + str(self.numStatesVisited) + os.linesep + \
                    'Terminal evaluations: ' + str(self.numTerminalStateEvals) + os.linesep + \
                    'Total paths traversed: ' + str(self.numPathsTraversed) + os.linesep + \
                    'Total paths to terminal node: ' + str(self.numPathsToTerminalNode) + os.linesep

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
            self.tree = self.Tree(state, self.sim, self.gd)

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
        self.evaluationAndBackPropagation(node.state, None, 1, 0)

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

        #node.prepareNode(self.sim, self.gd)

        turnTakers = node.turnTakers

        if len(turnTakers) == 0:
            #There will always be 1 child in this case as no one had a choice 
            #to move. Select that if explored, else return
            if len(node.children) == 0:
                if debug == 1:
                    print("Wierd case: no children" + str(state))
                return node
            else:
                childNode = next(iter(node.children))
                childNode.lastVisitedParent = node
                if debug == 1:
                    print("Wierd case: One children:" + str(state) + "-> "+ str(childNode.state))
                return childNode

        if len(turnTakers) == 1:
            #explore until all statistics are not available.
            if len(node.unexploredLegalMoves[turnTakers[0]]) != 0:
                if debug == 1:
                    print("Unexplored node. Will begin expansion and simulation" + str(state))
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
            if debug == 1:
                print("Terminal node reached: " + str(state))
            return node

        if self.role == turnTaker:
            maxReward = float("-inf")
            for child in node.children:
                reward = (child.avgReward) + \
                        100 * sqrt(2*log(node.allPlayed)/child.allPlayed)
                if debug == 1:
                    print("Bounds: " + str(reward) + " for state: " + str(child.state) + "Node played: " + str(node.allPlayed) + " child played: " + str(child.allPlayed))
                if reward > maxReward:
                    bestChild = child
                    maxReward = reward
        else:       #all opponents
            minReward = float("inf")
            for child in node.children:
                reward = (child.avgReward) - \
                        100 * sqrt(2*log(node.allPlayed)/child.allPlayed)
                if reward < minReward:
                    bestChild = child
                    minReward = reward
        if debug == 1:
            print("Selection : " + str(state) + "->" + str(bestChild.state) + " out of " + str([str(self.gd.moveTerm(x)) for x in node.unexploredLegalMoves[turnTaker]]))
        #this node would be the temporary parent of the child for this traversal.
        bestChild.lastVisitedParent = node
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
            node.unexploredLegalMoves = [list() for _ in xrange(len(moves))]      #empty list
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
        #node.prepareNode(self.sim, self.gd)
        # randomly select a node
        # Can implement light heuristics as well to choose nodes.
        if self.sim.isTerminal(state):
            if debug == 1:
                print("MCExpansion: Reached terminal state")
            return node

        nextState, moves = self.findNextUnexploredState(state)

        #if the state has already been traversed before, update whatever
        # rewards it has achieved till now and end the simulation process.
        nextNode = self.tree.findStateNode(nextState)
        if nextNode:
            self.tree.addChildNode(state, nextState, moves)
            nextNode.lastVisitedParent = node
            if debug == 1:
                print("Early return: " +str(state) + "->" + str(nextState) + " out of " + str([str(self.gd.moveTerm(x)) for x in node.unexploredLegalMoves[self.role]]))
            #return nextNode

        if nextState:
            self.tree.addChildNode(state, nextState, moves)
            nextNode = self.tree.findStateNode(nextState)
            nextNode.prepareNode(self.sim, self.gd)   #prepareNode made sense once, but not now.
            self.exploreStats.numStatesVisited += 1
            nextNode.lastVisitedParent = node
            if debug == 1:
                print("Expansion Printing state: " + str(state) + "->" +str(nextState) + " out of " + str([str(self.gd.moveTerm(x)) for x in node.unexploredLegalMoves[self.role]]))
        else:
            #The state had no unexplored move. Hence selection proceduce should 
            #be executed followed by expansion and simulation. It can also be 
            #the case that calls are made in reverse order which would be wrong.
            if debug == 1:
                print("Expansionn of explored node stopped, " + str(state))
            newNode = self.ucbSelection(state)
            nextState = newNode.state
            #raise ValueError("Call order of functions should be ucbSelecction->mcSimulationAndExpansion")
        return self.mcExpansionAndSimulation(nextState)

    def evaluationAndBackPropagation(self, state, playoutReward, playCount, level):
        node = self.tree.findStateNode(state)
        totalReward = node.avgReward * node.allPlayed
        if self.sim.isTerminal(state):
            #there is no point of dwelling over turntakers in terminal state.
            #No one can play in terminal state. its terminal.
            if not playoutReward:
                node.allPlayed += playCount
                if node.allPlayed == 1:     #incremented before, so 1
                    self.exploreStats.numTerminalStateEvals += 1
                if debug == 1:
                    print("Evaluation state: " + str(state))
                self.exploreStats.numPathsToTerminalNode += 1
                goals = self.sim.computeGoals(state)
                if debug == 1:
                    print("goals: " + str(goals))
                playoutReward = goals[self.role]
                node.avgReward = (totalReward + playoutReward)/node.allPlayed
            else:
                raise ValueError("Reward should be undefined only in case of terminal state.")
        else:
            if level != 0:
                node.allPlayed += playCount
                node.avgReward = (totalReward + playoutReward)/float(node.allPlayed)
            else:
                if debug == 1:
                    print("Middle Man: " + str(node.state) + " has reward: " + str(node.avgReward) + " and games at this node: " + str(node.allPlayed) + " and legalmoves " + str(len(node.legalMoves[self.role])) + " and move this time " + str(self.gd.moveTerm(node.lastVisitedParent.whichMove[node.state][self.role])) + "<-")
                return self.evaluationAndBackPropagation(node.lastVisitedParent.state, totalReward, node.allPlayed, level+1)

        if node.parents:
            if debug == 1:
                print(str(node.state) + " has reward: " + str(node.avgReward) + " and games at this node: " + str(node.allPlayed) + " and legal moves " + str(len(node.legalMoves[self.role])) + " and move this time " + str(self.gd.moveTerm(node.lastVisitedParent.whichMove[node.state][self.role])) + "<-")
            #for parent in node.parents:
            self.evaluationAndBackPropagation(node.lastVisitedParent.state, playoutReward, playCount, level+1)
        else:
            if debug == 1:
                print(str(node.state) + " has reward " + str(node.avgReward) + " and legal moves: " + str(len(node.legalMoves[self.role])))
                print("")
            pass
            #print(os.linesep)
