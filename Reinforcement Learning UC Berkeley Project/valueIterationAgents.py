# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
      for _ in range(self.iterations):
            nvals = self.values.copy()
            for s in self.mdp.getStates():
                if self.mdp.isTerminal(s):
                    nvals[s] = 0
                else:
                    qvs = []
                    for a in self.mdp.getPossibleActions(s):
                        qvs.append(self.computeQValueFromValues(s, a))
                    nvals[s] = max(qvs)
            self.values = nvals






    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
       tStates = self.mdp.getTransitionStatesAndProbs(state, action)
       Qv=0
       for nxstate, p in tStates:
           Qv += (p * (self.mdp.getReward(state, action, nxstate) + self.discount * self.values[nxstate]))
       return Qv


    def computeActionFromValues(self, state):
       if self.mdp.isTerminal(state):
           return None
       bestA = None
       bestV = None
       for a in self.mdp.getPossibleActions(state):
        qv = self.computeQValueFromValues(state,a)
        if bestV is None or bestV < qv:
                bestA = a
                bestV = qv
       return bestA


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        # Write value iteration code here
        for k in range(self.iterations):
            states = self.mdp.getStates()
            s = states[k % len(states)]
            nvals = self.values.copy()
            if self.mdp.isTerminal(s):
                    nvals[s] = 0
            else:
                    qvs = []
                    for a in self.mdp.getPossibleActions(s):
                        qvs.append(self.computeQValueFromValues(s, a))
                    nvals[s] = max(qvs)
            self.values = nvals

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()

        visitedStates = {}
        for state in states:
            visitedStates[state] = set()
        #possible actions from current state. At this step the states that have transition probs to the visited states dict.
        for state in states:
            for action in self.mdp.getPossibleActions(state):
                transitions = self.mdp.getTransitionStatesAndProbs(state, action)
                for trans in transitions:
                    visitedStates[trans[0]].add(state)

        priorityQ = util.PriorityQueue() #This will be used to get the maximum gain state

        for state in states:
            if self.mdp.isTerminal(state):
                continue
            bestAction = self.computeActionFromValues(state)
            bestQ = self.computeQValueFromValues(state, bestAction)
            priority = abs(bestQ - self.values[state])
            priorityQ.push(state, -priority)

        #calculating the reward and the maximum value for the priority states
        for iteration in range(self.iterations):
            if priorityQ.isEmpty():
                return
            state = priorityQ.pop()

            if not self.mdp.isTerminal(state):
                actions = self.mdp.getPossibleActions(state)
                maxVal = -696969
                for action in actions:
                    val = 0
                    nextStates = self.mdp.getTransitionStatesAndProbs(state, action)
                    for nextS in nextStates:
                        nextState = nextS[0]
                        nextProb = nextS[1]
                        reward = self.mdp.getReward(state, action, nextState)
                        r = self.values[nextState]
                        val += nextProb * (reward + self.discount * r)
                    maxVal = max(maxVal, val)
                if maxVal > -696969:
                    self.values[state] = maxVal
                else:
                    self.values[state] = 0

            #Now, we can choose the best action that has the highest Q value.
            for v in visitedStates[state]:
                bestAction = self.computeActionFromValues(v)
                if bestAction == None:
                    continue
                bestQ = self.computeQValueFromValues(v, bestAction)
                newPriority = abs(bestQ - self.values[v])

                if newPriority > self.theta:
                    priorityQ.update(v, -newPriority)

