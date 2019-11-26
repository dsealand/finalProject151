from tilefeatures import *
from rl import *
import sys
import random
import argparse
import math
turtle = None #placeholder for turtle module

class Function:    
    def getValue(self, x, y):
        return x*y

class MountainCar:
    '''Represents the Mountain Car problem.'''
    def __init__(self):
        self.reset()

        # self.__xEnd = random.random() * 10
        # self.__yEnd = random.random() * 10 

    def reset(self):
        '''Resets the problem to the initial state.'''                
        self.__xPos = random.random() * 10
        self.__yPos = random.random() * 10

        self.__xEnd = random.random() * 10
        self.__yEnd = random.random() * 10
        
    def transition(self, action):
        '''Transitions to the next state, depending on the action. Actions 0, 1, and 2 are reverse, neutral, and forward, respectively. Returns the reward from the transition (always -1).'''        
        if self.isTerminal():
            return 0
        
        if action not in range(360):
            raise ValueError("Invalid action: " + str(action))

        actionRadian = math.radians(action)

        surface = Function()
        initialCost = surface.getValue(self.__xPos, self.__yPos)

        self.__xPos += 0.1 * math.cos(actionRadian)
        self.__yPos += 0.1 * math.sin(actionRadian)

        # print(self.__xPos)
        # print(self.__yPos)

        if self.__xPos > 10:
            self.__xPos = 10
        elif self.__xPos < 0:
            self.__xPos = 0

        if self.__yPos > 10:
            self.__yPos = 10
        elif self.__yPos < 0:
            self.__yPos = 0

        cost = surface.getValue(self.__xPos, self.__yPos) - initialCost
        if cost > 0:
            return -1 - cost
        return -1

    def isTerminal(self):
        '''Returns true if the world is in a terminal state (if the car is at the top of the hill).'''
        return math.fabs(self.__xPos - self.__xEnd) < 1 and math.fabs(self.__yPos - self.__yEnd) < 1

    def getState(self):
        '''Returns a tuple containing the position and velocity of the car, in that order.'''
        return (self.__xPos, self.__yPos, self.__xEnd, self.__yEnd)

    def getInitialState(self):
        return (self.__xPos, self.__yPos)

    def getTerminalState(self):
        return (self.__xEnd, self.__yEnd)

    def setTerminalState(self, x, y):
        self.__xEnd = x
        self.__yEnd = y

    def setInitialState(self, x, y):
        self.__xPos = x
        self.__yPos = y

    def getRanges(self):
        '''Returns a tuple of lists representing the ranges of the two state variables. There are two lists of two elements each, the minimum and maximum value, respectively.'''
        return ([0, 10], [0, 10], [0,10], [0,10])
    
    def __str__(self):
        return "xpos: " + str(self.__xPos) + "ypos: " + str(self.__yPos)

class MountainCarDisplay:
    '''Uses turtle to visualize the Mountain Car problem.'''
    def __init__(self, world):
        '''Takes a MountainCar object and initializes the display.'''
        self.__world = world

        surface = Function()

        #Create the window
        turtle.setup(800, 800)
        turtle.setworldcoordinates(0, 0, 10, 10)
        turtle.title("3d route finding")
        turtle.bgcolor(1, 1, 1)
        turtle.tracer(0)
        turtle.colormode(255)

        # draw surface
        hillT = turtle.Turtle()
        hillT.hideturtle()
        hillT.penup()
        x = 0
        y = 0
        hillT.penup()
        hillT.goto(x, y)
        # change color based on elevation
        hillT.pencolor(0, 0, 0)
        hillT.pensize(10)
        hillT.begin_fill()
        while x < 10:
            while y < 10:
                functionValue = surface.getValue(x, y)
                hillT.pencolor(0, 0, int(2.5*functionValue))
                hillT.pendown()
                hillT.goto(x, y)
                
                hillT.penup()
                
                
                y += 0.1
            x += 0.1
            y = 0
            hillT.pencolor(0, 0, int(2.5*functionValue))
            hillT.goto(x, y)
        hillT.end_fill()


        #Create the car turtle
        self.__carTurtle = turtle.Turtle()
        self.__carTurtle.shape("circle")
        self.__carTurtle.shapesize(1.5, 1.5, 3)
        self.__carTurtle.fillcolor(255, 0, 0)
        self.__carTurtle.pencolor(0, 255, 0)
        self.__carTurtle.pendown()

        self.update()
            
    def update(self):
        '''Updates the display to reflect the current state.'''
        state = self.__world.getState()
        self.__carTurtle.goto(state[0], state[1])

        turtle.update()

    def exitOnClick(self):
        turtle.exitonclick()

def main():
    parser = argparse.ArgumentParser(description='Use Sarsa with linear value function approximation to solve the Mountain Car problem.')
    parser.add_argument('output_file', help='file to output the learning data')
    parser.add_argument('-a', '--alpha', type=float, default=0.1, help='the step-size alpha (default: 0.1)')
    parser.add_argument('-e', '--epsilon', type=float, default=0.1, help='the exploration rate epsilon (default: 0.1)')
    parser.add_argument('-g', '--gamma', type=float, default=0.9, help='the discount factor gamma (default: 0.9)')
    parser.add_argument('-t', '--trials', type=int, default=1, help='the number of trials to run (default: 1)')
    parser.add_argument('-p', '--episodes', type=int, default=200, help='the number of episodes per trial (default: 200)')
    parser.add_argument('-m', '--maxsteps', type=int, default=2000, help="the maximum number of steps per episode (default: 2000)")    
    parser.add_argument('-d', '--display', metavar="N", type=int, default=0, help='display every Nth episode (has no effect if TRIALS > 1)')
    parser.add_argument('-n', '--numtilings', type=int, default=5, help='the number of tilings to use')
    parser.add_argument('-s', '--numtiles', metavar="N", type=int, default=9, help='each tiling will divide the space into an NxN grid')

    args = parser.parse_args()
    fout = open(args.output_file, "w")
    world = MountainCar()

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

    if args.display > 0 and not displayError:
        try:
            display = MountainCarDisplay(world)
        except Exception as ex:
            args.display = 0
            print("ERROR: " + str(ex))
            print("WARNING: Unable to initialize the GUI. Display will be disabled.")
    else:
        args.display = 0
        
    avgTotal = [0]*args.episodes
    avgDiscounted = [0]*args.episodes
    avgSteps = [0]*args.episodes
    for trial in range(args.trials):
        if args.trials > 1:
            print("Trial " + str(trial+1), end="")        
        featureGenerator = TileFeatures(world.getRanges(), [args.numtiles, args.numtiles, args.numtiles, args.numtiles], args.numtilings)
        agent = LinearSarsaLearner(featureGenerator.getNumFeatures(), 360, args.alpha, args.epsilon, args.gamma)

        repeatGoals = [0] * 10
        repeatStarts = [0] * 10
        for ep in range(args.episodes):
            if args.display > 0:
                displayEp = ep == 0 or (ep+1)%args.display == 0
            else:
                displayEp = False
                
            if args.trials > 1:
                print(".", end="", flush=True)
            totalR = 0
            discountedR = 0
            discount = 1
            world.reset()


            if ep < 10:
                repeatGoals[ep] = world.getTerminalState()
                repeatStarts[ep] = world.getInitialState()

            if args.episodes - ep < 10:
                xEnd = repeatGoals[args.episodes - ep][0]
                yEnd = repeatGoals[args.episodes - ep][1]
                world.setTerminalState(xEnd,yEnd)

                x = repeatStarts[args.episodes - ep][0]
                y = repeatStarts[args.episodes - ep][1]
                world.setInitialState(x,y)

            activeFeatures = featureGenerator.getFeatures(world.getState())
            action = agent.epsilonGreedy(activeFeatures)
            reward = world.transition(action)
            if displayEp:
                display.update()
            totalR += reward
            discountedR += discount*reward            
            step = 1
            while not world.isTerminal() and step < args.maxsteps:
                newFeatures = featureGenerator.getFeatures(world.getState())             
                action = agent.learningStep(activeFeatures, action, reward, newFeatures)
                activeFeatures = newFeatures
                reward = world.transition(action)
                if displayEp:
                    display.update()
                totalR += reward
                discount *= args.gamma
                discountedR += discount*reward
                step += 1
            if world.isTerminal():
                agent.terminalStep(activeFeatures, action, reward)
            else:
                newFeatures = featureGenerator.getFeatures(world.getState())                
                agent.learningStep(activeFeatures, action, reward, newFeatures)
            avgTotal[ep] += totalR
            avgDiscounted[ep] += discountedR
            avgSteps[ep] += step
            if args.trials == 1:
                print("Episode " + str(ep+1) + ": " + str(totalR) + " " + str(discountedR) + " " + str(step))
        if args.trials > 1:
            print("")
           
    for i in range(args.episodes):
        avgStr = str(avgTotal[i]/args.trials) + " " + str(avgDiscounted[i]/args.trials) + " " + str(avgSteps[i]/args.trials)
        fout.write(str(i+1) + " " + avgStr + "\n")
        if args.trials > 1:
            print("Average episode " + str(i) + ": " + avgStr)

    if args.display > 0:
        print("Click the display window to exit")
        display.exitOnClick()
        
if __name__ == "__main__":
    main()
