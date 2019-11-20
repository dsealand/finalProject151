'''
The rl module contains multiple reinforcement learning algorithms
'''

import random
from tilefeatures import *
import abc
import math


class QTableLearner(metaclass=abc.ABCMeta):
    '''Represents an abstract reinforcement learning agent that represents its Q function using a look-up table.'''
    def __init__(self, numStates, numActions, alpha, epsilon, gamma, initQ):
        '''The constructor takes the number of states and actions in the MDP as well as the step size (alpha), the exploration rate (epsilon), the discount factor (gamma), and the initial Q-value for all state-action pairs.'''
        self.q = [] #q is indexed first by state, then by action
        for s in range(numStates):
            self.q.append([initQ]*numActions)

        self.numStates = numStates
        self.numActions = numActions
            
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
    
    def epsilonGreedy(self, state):
        '''With probability epsilon returns a uniform random action. Otherwise it returns a greedy action with respect to the current Q function (breaking ties randomly).'''
        randomFloat = random.random()
        if randomFloat <= self.epsilon:
            output = random.choice([*range(self.numActions)])
        else:
            output = self.greedy(state)
        return output
                            
    def greedy(self, state):
        '''Returns a greedy action with respect to the current Q function (breaking ties randomly).'''
        actions = self.q[state]
        maxQ = -math.inf
        maxI = 0
        for i in [*range(len(actions))]:
            if actions[i] > maxQ:
                maxQ = actions[i]
                maxI = i
        return maxI

    def terminalStep(self, curState, action, reward):
        '''Performs the last learning step of an episode. Because the episode has terminated, the next Q-value is 0.'''
        currentQ = self.q[curState][action]
        newQ = currentQ + self.alpha * (reward - currentQ)
        self.q[curState][action] = newQ
        return 0
        
    @abc.abstractmethod
    def learningStep(self, curState, action, reward, nextState):
        '''Performs a learning step based on the given transition. Returns the action the agent will take next.'''
        pass
        
class SarsaLearner(QTableLearner):
    '''Represents an agent using the SARSA algorithm.'''
    def __init__(self, numStates, numActions, alpha, epsilon, gamma, initQ=0):
        '''The constructor takes the number of states and actions in the MDP as well as the step size (alpha), the exploration rate (epsilon), the discount factor (gamma), and the initial Q-value for all state-action pairs (0 by default).'''
        super().__init__(numStates, numActions, alpha, epsilon, gamma, initQ)
        
    def learningStep(self, curState, action, reward, nextState):
        '''Performs a SARSA learning step based on the given transition. Returns the action the agent will take next.'''
        nextAction = self.epsilonGreedy(curState)
        currentQ = self.q[curState][action]
        nextQ = currentQ + self.alpha * (reward + self.gamma * self.q[nextState][nextAction] - currentQ)
        self.q[curState][action] = nextQ
        return nextAction
                
class QLearner(QTableLearner):
    '''Represents an agent using the Q-learning algorithm.'''
    def __init__(self, numStates, numActions, alpha, epsilon, gamma, initQ=0):
        '''The constructor takes the number of states and actions in the MDP as well as the step size (alpha), the exploration rate (epsilon), the discount factor (gamma), and the initial Q-value for all state-action pairs (0 by default).'''
        super().__init__(numStates, numActions, alpha, epsilon, gamma, initQ)
        
    def learningStep(self, curState, action, reward, nextState):
        '''Performs a Q-learning step based on the given transition. Returns the action the agent will take next.'''
        nextAction = self.greedy(nextState)
        currentQ = self.q[curState][action]
        nextQ = currentQ + self.alpha * (reward + self.gamma * self.q[nextState][nextAction] - currentQ)
        self.q[curState][action] = nextQ
        return nextAction

class LinearSarsaLearner:
    '''Represents an agent using SARSA with linear value function approximation, assuming binary features.'''
    def __init__(self, numFeatures, numActions, alpha, epsilon, gamma):
        '''The constructor takes the number of features and actions as well as the step size (alpha), the exploration rate (epsilon), the discount factor (gamma).'''
        self.theta = [] #theta represent the weights of the Q function. It is indexed first by action, then by feature index
        for a in range(numActions):
            self.theta.append([0]*numFeatures)

        self.numFeatures = numFeatures
        self.numActions = numActions
            
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

    def getQValue(self, activeFeatures, action):
        '''Calculates the approximate Q-value of a state-action pair. It takes a list of indices of active features (feature value is 1) and the action.'''
        qValue = 0
        for feature in activeFeatures:
            qValue += self.theta[action][feature]
        return qValue
                    
    def epsilonGreedy(self, activeFeatures):
        '''With probability epsilon returns a uniform random action. Otherwise it returns a greedy action with respect to the current Q function (breaking ties randomly).'''
        randomFloat = random.random()
        if randomFloat <= self.epsilon:
            output = random.choice(range(self.numActions))
        else:
            output = self.greedy(activeFeatures)
        return output

    def greedy(self, activeFeatures):
        '''Returns a greedy action with respect to the current Q function (breaking ties randomly).'''
        maxQ = -math.inf
        maxI = None
        for i in range(self.numActions):
            qValue = self.getQValue(activeFeatures, i)
            if qValue > maxQ:
                maxQ = qValue
                maxI = i
            elif qValue == maxQ:
                x = random.randint(0, 1)
                if x == 1:
                    maxI = i
        return maxI
        
    def learningStep(self, activeFeatures, action, reward, nextFeatures):
        '''Performs a gradient descent SARSA learning step based on the given transition. Returns the action the agent will take next.'''
        nextAction = self.epsilonGreedy(nextFeatures)
        currentQ = self.getQValue(activeFeatures, action)
        delta = reward + self.gamma * self.getQValue(nextFeatures, nextAction) - currentQ
        for i in activeFeatures:
            self.theta[action][i] = self.theta[action][i] + self.alpha * delta
        return nextAction
                        
    def terminalStep(self, activeFeatures, action, reward):
        '''Performs the last learning step of an episode. Because the episode has terminated, the next Q-value is 0.'''
        currentQ = self.getQValue(activeFeatures, action)
        delta = reward - currentQ
        for i in activeFeatures:
            self.theta[action][i] = self.theta[action][i] + self.alpha * delta
        