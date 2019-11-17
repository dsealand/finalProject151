from gridworld import *
from rl import *
import sys
import random
import argparse
import time
import math
turtle = None #placeholder for turtle module

class GridWorld:
    '''Represents a grid world problem.'''
    def __init__(self, filename):
        '''Takes the name of a file that contains the layout of the grid. The first line of the file should contains the dimensions of the grid: number of rows and number of columns, in that order. Then each entry in the grid should have three comma-separated elements:
          - One of *, ., or # to represent the start state, a blank space, or a wall, respectively
          - The reward for entering that square
          - Either T or F to indicate whether the state is terminal'''
        fin = open(filename)
        self.__dims = [int(d) for d in fin.readline().split()]

        self.__grid = []
        self.__curState = None

        self.__maxR = -float("inf")
        self.__minR = float("inf")        
        
        for i in range(self.__dims[0]):
            self.__grid.append([])
            tokens = fin.readline().split()

            if len(tokens) != self.__dims[1]:
                raise ValueError("Length of row " + str(i) + " does not match given width: " + str(len(tokens)) + " (expected " + str(self.__dims[1]) + ")")
            
            for j in range(len(tokens)):
                attributes = tokens[j].split(",") # Wall or not, reward, terminal or not

                if len(attributes) != 3:
                    raise ValueError("Unexpected number of attributes for position " + str((i, j)) + ": " + str(len(attributes)) + " (expected 3 comma-separated values)")
                
                if attributes[0] not in [".", "#", "*"]:
                    raise ValueError("Unexpected layout symbol for position " + str((i, j)) + ": " + attributes[0] + " (expected ., #, or *)")

                if attributes[0] == "*": #Starting location
                    if self.__curState != None:
                        raise ValueError("Multiple start states detected at " + str(self.__curState) + " and " + str((i, j)))
            
                    self.__curState = (i, j)
                    attributes[0] = "."

                attributes[1] = float(attributes[1])
                self.__maxR = max(attributes[1], self.__maxR)
                self.__minR = min(attributes[1], self.__minR)                

                if attributes[2] == "T":
                    attributes[2] = True
                elif attributes[2] == "F":
                    attributes[2] = False
                else:
                    raise ValueError("Unexpected terminal indicator at position " + str((i, j)) + ": " + attributes[2] + " (expected T or F)")
                    
                self.__grid[-1].append(attributes)

        if self.__curState == None:
            raise ValueError("No start state detected.")
        else:
            self.__initState = self.__curState

        self.__actionMap = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def reset(self):
        '''Resets the problem to the initial state.'''
        self.__curState = self.__initState
        
    def transition(self, action):
        '''Moves the agent in the grid. The actions represent the cardinal directions. Actions 0, 1, 2, and 3 represent North, East, South, and West, respectively. Trying to move onto a wall or off the grid results in no movement. Returns the reward from the transition.'''
        if self.isTerminal():
            return 0

        if action not in range(4):
            raise ValueError("Invalid action: " + str(action))
        
        direction = self.__actionMap[action]

        newLoc = (self.__curState[0]+direction[0], self.__curState[1]+direction[1])

        if self.isInBounds(newLoc) and not self.isWall(newLoc):
            self.__curState = newLoc

        return self.getReward(self.__curState)

    def getNumStates(self):        
        return self.__dims[0]*self.__dims[1]
    
    def getState(self):
        '''Gets the current state as a number between 0 and width*height-1.'''
        return self.__curState[0]*self.__dims[1] + self.__curState[1]

    def getAgentLoc(self):
        '''Returns the current state as a tuple: (row, column).'''
        return self.__curState[0], self.__curState[1]
        
    def __str__(self):
        stateStr = ""
        for i in range(len(self.__grid)):
            for j in range(len(self.__grid[i])):
                if (i, j) == self.__curState:
                    stateStr += "*"
                else:
                    stateStr += self.__grid[i][j][0]
            stateStr += "\n"
        return stateStr                
    
    def isInBounds(self, loc):
        '''Determines if a given location (row, column) is inside the grid.'''
        return loc[0] >= 0 and loc[0] < self.__dims[0] and loc[1] >= 0 and loc[1] < self.__dims[1]

    def isTerminal(self):
        '''Returns True if the environment is currently in a terminal state.'''
        return self.isTerminalLoc(self.__curState)
    
    def isTerminalLoc(self, loc):
        '''Determines if a given location (row, column) is terminal.'''        
        return self.__grid[loc[0]][loc[1]][2]

    def isWall(self, loc):
        '''Determines if a given location (row, column) is a wall.'''        
        return self.__grid[loc[0]][loc[1]][0] == "#"

    def getReward(self, loc):
        '''Returns the reward associated with the given location (row, column).'''        
        return self.__grid[loc[0]][loc[1]][1]

    def getDims(self):
        '''Returns a tuple containing the number of rows and number of columns in the grid.'''
        return tuple(self.__dims)

    def getMaxReward(self):
        '''Returns the maximum reward value in the grid.'''
        return self.__maxR

    def getMinReward(self):
        '''Returns the minimum reward value in the grid.'''        
        return self.__minR

class GridWorldDisplay:
    '''Uses turtles to visualize the grid world and the Q-function.'''
    def __init__(self, world, bigQ):
        '''Takes a GridWorld and a value representing the largest magnitude a Q-value can take on (this is to scale the coloring of the Q-function visualization.'''
        self.__world = world
        numR, numC = world.getDims()

        bigReward = max(abs(world.getMaxReward()), abs(world.getMinReward()))
        self.__bigQ = bigQ

        #Create the window
        squareWidth = 50
        turtle.setup(numC*squareWidth+10, numR*squareWidth+16)
        turtle.setworldcoordinates(0, numR+10/squareWidth, numC+10/squareWidth, 0)
        turtle.title("Grid World")
        turtle.tracer(0)

        self.__headings = [270, 0, 90, 180]
        self.__triangleSide = 0.3
        self.__triangleHyp = self.__triangleSide*math.sqrt(2)
        
        #Draw the grid
        self.__gridT = turtle.Turtle()
        self.__gridT.hideturtle()
        self.__gridT.pencolor(0, 0, 0)

        for r in range(numR):
            for c in range(numC):
                self.__gridT.width(3)
                self.__gridT.goto(c, r)
                if world.isWall((r, c)):
                    self.__gridT.fillcolor(0, 0, 0)
                elif world.getReward((r, c)) >= 0:                    
                    self.__gridT.fillcolor(1 - world.getReward((r, c))/bigReward, 1, 1 - world.getReward((r, c))/bigReward)
                else: #reward < 0
                    self.__gridT.fillcolor(1, 1 + world.getReward((r, c))/bigReward, 1 + world.getReward((r, c))/bigReward)
                                        
                self.__gridT.begin_fill()                
                self.__gridT.pendown()
                for i in range(4):
                    self.__gridT.forward(1)
                    self.__gridT.left(90)
                self.__gridT.end_fill()
                self.__gridT.penup()

                if world.isTerminalLoc((r, c)):
                    self.__gridT.goto(c+0.1, r+0.1)
                    self.__gridT.pendown()
                    for i in range(4):
                        self.__gridT.forward(0.8)
                        self.__gridT.left(90)
                    self.__gridT.penup()

        #Create the turtle that draws the Q-function
        self.__triangleT = turtle.Turtle()
        self.__triangleT.hideturtle()
        self.__triangleT.pencolor(0, 0, 0)
        self.__triangleT.width(1)
        self.__triangleT.penup()

        #Create the agent turtle
        self.__playerTurtle = turtle.Turtle()
        self.__playerTurtle.shape("circle")
        self.__playerTurtle.shapesize(1.5, 1.5, 3)
        self.__playerTurtle.fillcolor(0, 0, 1)
        self.__playerTurtle.pencolor(0, 0, 0)
        self.__playerTurtle.penup()        

    def update(self, q):
        '''Updates the display to repsent the current state and also the given Q function (assumed to be a list of lists of Q-values, indexed by state and then action.'''
        #First update the Q-function visualization
        self.__triangleT.clear()
        numR, numC = self.__world.getDims()
        for r in range(numR):
            for c in range(numC):        
                if not self.__world.isTerminalLoc((r, c)) and not self.__world.isWall((r, c)):      
                    self.__triangleT.width(1)
                    state = r*numC + c
                    maxQ = max(q[state])
                    greedyA = [a for a in range(4) if q[state][a] == maxQ]
                                           
                    for a in range(4):
                        qVal = q[r*numC+c][a]
                        if qVal >= 0:
                            self.__triangleT.fillcolor(1 - qVal/self.__bigQ, 1, 1 - qVal/self.__bigQ)
                        else: #qVal < 0
                            self.__triangleT.fillcolor(1, 1 + qVal/self.__bigQ, 1 + qVal/self.__bigQ)
                        if a in greedyA:
                            self.__triangleT.width(2)
                        else:
                            self.__triangleT.width(1)
                            
                        self.__triangleT.goto(c+0.5, r+0.5)
                        self.__triangleT.setheading(self.__headings[a])
                        self.__triangleT.forward(0.45)
                        self.__triangleT.right(135)
                        self.__triangleT.pendown()
                        self.__triangleT.begin_fill()
                        self.__triangleT.forward(self.__triangleSide)
                        self.__triangleT.right(135)
                        self.__triangleT.forward(self.__triangleHyp)
                        self.__triangleT.right(135)
                        self.__triangleT.forward(self.__triangleSide)
                        self.__triangleT.end_fill()                        
                        self.__triangleT.penup()

        #Update the agent's position
        playerRow, playerCol = self.__world.getAgentLoc()
        self.__playerTurtle.goto(playerCol+0.5, playerRow+0.5)
        turtle.update()

        #Pause briefly to make the animation bearable
        time.sleep(0.1)
                
    def exitOnClick(self):
        turtle.exitonclick()

def main():
    parser = argparse.ArgumentParser(description='Use reinforcement learning algorithms to solve gridworld problems.')
    parser.add_argument('grid_file', help='file containing the grid layout')
    parser.add_argument('output_file', help='file to output the learning data')
    parser.add_argument('-l', '--learner', type=str, choices=['q', 'sarsa'], default="q", help="the learning algorithm to use (default: q)")
    parser.add_argument('-a', '--alpha', type=float, default=0.1, help='the step-size alpha (default: 0.1)')
    parser.add_argument('-e', '--epsilon', type=float, default=0.1, help='the exploration rate epsilon (default: 0.1)')
    parser.add_argument('-g', '--gamma', type=float, default=0.9, help='the discount factor gamma (default: 0.9)')
    parser.add_argument('-i', '--initQ', type=float, default=0, help='the initial Q-value')
    parser.add_argument('-t', '--trials', type=int, default=1, help='the number of trials to run')
    parser.add_argument('-p', '--episodes', type=int, default=500, help='the number of episodes per trial')
    parser.add_argument('-m', '--maxsteps', type=int, default=100, help='the maximum number of steps per episode (default: 100)')
    parser.add_argument('-d', '--display', metavar="N", type=int, default=0, help='display every Nth episode (has no effect if TRIALS > 1)')
    parser.add_argument('-b', '--bigQ', type=float, help='the biggest possible magnitude for Q-values for display purposes (default: the largest reward value)')


    args = parser.parse_args()
    fout = open(args.output_file, "w")
    world = GridWorld(args.grid_file)

    displayError = False
    if args.display > 0:
        try:
            global turtle
            import turtle
        except Exception as ex:
            displayError = True
            print("ERROR: " + str(ex))
            print("WARNING: Unable to initialize the GUI. Display will be disabled.")    
    if args.trials != 1:
        args.display = 0

    if args.bigQ == None:
        args.bigQ = max(abs(world.getMaxReward()), abs(world.getMinReward()))
            
    if args.display > 0 and not displayError:
        try:
            display = GridWorldDisplay(world, args.bigQ)
        except Exception as ex:
            args.display = 0
            print("ERROR: " + str(ex))
            print("WARNING: Unable to initialize the GUI. Display will be disabled.")
    else:
        args.display = 0
    
    avgTotal = [0]*args.episodes
    avgDiscounted = [0]*args.episodes
    avgSteps = [0]*args.episodes
    avgFinalTotal = 0
    avgFinalDiscounted = 0
    avgFinalSteps = 0
    for trial in range(args.trials):
        if args.trials > 1:
            print("Trial " + str(trial+1), end=" ")
        if args.learner == "sarsa":
            agent = SarsaLearner(world.getNumStates(), 4, args.alpha, args.epsilon, args.gamma, args.initQ)
        elif args.learner == "q":
            agent = QLearner(world.getNumStates(), 4, args.alpha, args.epsilon, args.gamma, args.initQ)
        else:
            raise ValueError("Unknown learner type: " + args.learner)

        for ep in range(args.episodes):            
            if args.display > 0:
                displayEp = ep == 0 or (ep+1)%args.display == 0
            else:
                displayEp = False

            totalR = 0
            discountedR = 0
            discount = 1
            world.reset()
            if displayEp:
                display.update(agent.q)                
            curState = world.getState()
            action = agent.epsilonGreedy(curState)
            reward = world.transition(action)
            totalR += reward
            discountedR += discount*reward
            step = 1
            while not world.isTerminal() and step < args.maxsteps:
                action = agent.learningStep(curState, action, reward, world.getState())
                if displayEp:
                    display.update(agent.q)
                curState = world.getState()
                reward = world.transition(action)
                totalR += reward
                discount *= args.gamma
                discountedR += discount*reward
                step += 1
            if world.isTerminal():
                agent.terminalStep(curState, action, reward)
            else:
                agent.learningStep(curState, action, reward, world.getState())
            if displayEp:
                display.update(agent.q)
            if args.trials == 1:
                print("Episode " + str(ep+1) + ": " + str(totalR) + " " + str(discountedR) + " " + str(step))
            avgTotal[ep] += totalR
            avgDiscounted[ep] += discountedR
            avgSteps[ep] += step

        world.reset()
        if args.display > 0:
            display.update(agent.q)                        
        totalR = 0
        discountedR = 0
        discount = 1
        curState = world.getState()
        action = agent.greedy(curState)
        reward = world.transition(action)
        totalR += reward
        discountedR += discount*reward
        step = 1
        while not world.isTerminal() and step < args.maxsteps:  
            action = agent.greedy(world.getState())
            if args.display > 0:
                display.update(agent.q)                        
            curState = world.getState()
            reward = world.transition(action)
            totalR += reward
            discount *= args.gamma
            discountedR += discount*reward
            step += 1
        if args.display > 0:
            display.update(agent.q)                        
        print("Final greedy policy: " + str(totalR) + " " + str(discountedR) + " " + str(step), flush=True)

        avgFinalTotal += totalR
        avgFinalDiscounted += discountedR
        avgFinalSteps += step
                
    for i in range(args.episodes):
        avgStr = str(avgTotal[i]/args.trials) + " " + str(avgDiscounted[i]/args.trials) + " " + str(avgSteps[i]/args.trials)
        fout.write(str(i+1) + " " + avgStr + "\n")
        if args.trials > 1:
            print("Average episode " + str(i+1) + ": " + avgStr)

    if args.trials > 1:
        print("Average final greedy policy: " + str(avgFinalTotal/args.trials) + " " + str(avgFinalDiscounted/args.trials) + " " + str(avgFinalSteps/args.trials))
        
    if args.display > 0:
        print("Click the display window to exit")
        display.exitOnClick()
        
if __name__ == "__main__":
    main()
