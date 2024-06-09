# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
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

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

from __future__ import absolute_import
from __future__ import print_function

import math
import random

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util


class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm

    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """

    def __init__(self, state: GameState):
        """
        Args:
            state: A given game state object
        """
        # Extract useful information from the game state
        self.pacman = state.getPacmanPosition()
        # Convert ghost position(s) to a tuple to make the state hashable
        self.ghosts = tuple(state.getGhostPositions())
        self.food = state.getFood()
        self.legalActions = state.getLegalPacmanActions()
        # Flag to indicate if state is a terminal state
        self.final = False

    # Check if another state is equal based on its position (pacman),
    # ghost (ghost position(s)), and food (food locations)
    def __eq__(self, other):
        return self.pacman == other.pacman and self.ghosts == other.ghosts and self.food == other.food

    # Encode a state based on its position (pacman), ghost (ghost position(s)), and food (food locations)
    def __hash__(self):
        return hash((self.pacman, self.ghosts, self.food))


class QLearnAgent(Agent):

    def __init__(self,
                 alpha: float = 0.2,
                 epsilon: float = 0.05,
                 gamma: float = 0.8,
                 maxAttempts: int = 30,
                 numTraining: int = 10):
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        The given hyperparameters are suggestions and are not necessarily optimal
        so feel free to experiment with them.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0
        # Initialise Q-values and state-action pair counts dictionaries
        self.qValues = {}
        self.actionCounts = {}
        # Initialise previous state (GameState object), previous state (GameStateFeatures object),
        # previous action, and previous reward
        self.prevState = None
        self.prevStateFeatures = None
        self.prevAction = None
        self.prevReward = None

    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    @staticmethod
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """

        # Reward is the difference in score between two successive states
        reward = endState.getScore() - startState.getScore()
        return reward

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """
        # Return the Q-value for the given state-action pair, or 0 if it doesn't exist
        return self.qValues.get((state, action), 0)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        # Initialise list to store all Q-values of the state with each action
        q_values = []

        for action in state.legalActions:
            q_s_a = self.getQValue(state, action)
            q_values.append(q_s_a)
        # If no Q-values exist for the state, return 0
        if not q_values:
            return 0
        # Return the maximum Q-value for the given state
        return max(q_values)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        """
        Performs a Q-learning update

        Args:
            state: the initial state
            action: the action that was taken
            nextState: the resulting state
            reward: the reward received on this trajectory
        """
        # Get Q-value for state-action pair
        q_s_a = self.getQValue(state, action)

        # If nextState is a terminal state, set max_q_sp_ap to 0, as there are
        # no future actions to take from this terminal state
        if nextState.final:
            max_q_sp_ap = 0
        else:  # Not a terminal state
            # Get maximum Q-value for next state (s') based on actions s' can take (a')
            max_q_sp_ap = self.maxQValue(nextState)

        # Decay learning rate based on state-action pair count
        alpha = self.alpha / (1 + (self.getCount(state, action) / 100))

        # Update Q-value, Q(s,a) ← Q(s,a) + α*(r + γ*max(Q(s',a')) − Q(s,a))
        new_q_s_a = q_s_a + alpha * (reward + self.gamma * max_q_sp_ap - q_s_a)
        self.qValues[(state, action)] = new_q_s_a

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        # If state-action pair exists in count dictionary, increment count
        if (state, action) in self.actionCounts:
            self.actionCounts[(state, action)] += 1
        else:  # If state-action pair not in dictionary, add it with count 1
            self.actionCounts[(state, action)] = 1

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
        """

        # Return count for state-action pair. If it doesn't exist, return 0
        return self.actionCounts.get((state, action), 0)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def explorationFn(self,
                      utility: float,
                      counts: int) -> float:
        """
        Computes exploration function.
        Return a value based on the counts

        HINT: Do a greed-pick or a least-pick

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited

        Returns:
            The exploration value
        """
        # If in the training phase, use the exploration function
        if self.epsilon > 0:
            # Normalize Q-value to a value between 0 and 1 using the sigmoid function
            # to not punish small negative Q-values too much
            sigmoid_utility = 1 / (1 + math.exp(-utility))

            # Divide by count to promote exploration for less-visited state-action pairs
            if counts > 0:
                return sigmoid_utility / counts
            else:
                return sigmoid_utility  # No division by zero
        else:  # When learning is done, don't factor in number of visits
            return utility

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning

        If you wish to use epsilon-greedy exploration, implement it in this method.
        HINT: look at pacman_utils.util.flipCoin

        Args:
            state: the current state

        Returns:
            The action to take
        """
        currentStateFeatures = GameStateFeatures(state)
        currentLegalActions = currentStateFeatures.legalActions

        # Remove Directions.STOP from legalActions if it exists
        currentLegalActions = [action for action in currentLegalActions if action != Directions.STOP]

        # epsilon-greedy exploration
        if util.flipCoin(self.epsilon):
            chosen_action = random.choice(currentLegalActions)
        else:
            # Initialise variables to store the best actions for current state and their utility values
            bestActions = []
            maxUtility = float('-inf')

            # Find best action from current state, taking into account the exploration function (explorationFn)
            for action in currentLegalActions:
                q_s_a = self.getQValue(currentStateFeatures, action)
                count_s_a = self.getCount(currentStateFeatures, action)
                # Calculate the utility of the state-action pair based of explorationFn
                utility = self.explorationFn(q_s_a, count_s_a)

                # Find the max utility action(s)
                if utility > maxUtility:
                    bestActions = [action]
                    maxUtility = utility
                elif utility == maxUtility:
                    bestActions.append(action)

            # Choose randomly among the best actions (if only 1 best action, then it will be the chosen one)
            # This is the action that will be performed
            chosen_action = random.choice(bestActions)

        # Update (increment) the count for the chosen state-action pair
        self.updateCount(currentStateFeatures, chosen_action)

        # If not the first step, perform learning step using the previous state,
        # previous action, reward from previous state to current state, and current state.
        # Learning is done one step after the action is chosen because we can't access
        # the next state object until the action is taken.
        if self.prevState is not None:
            # Store reward for previous state moving into current state
            self.prevReward = self.computeReward(self.prevState, state)
            # Perform learning update
            self.learn(self.prevStateFeatures, self.prevAction, self.prevReward, currentStateFeatures)

        # Store current state and action for next learning step
        self.prevAction = chosen_action
        self.prevState = state  # Current state (GameState object)
        self.prevStateFeatures = currentStateFeatures  # Current state (GameStateFeatures object)

        return chosen_action

    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """
        print(f"Game {self.getEpisodesSoFar()} just ended!")

        finalStateFeatures = GameStateFeatures(state)
        # Final state of game flag on
        finalStateFeatures.final = True

        # Perform the final learning update
        self.learn(self.prevStateFeatures, self.prevAction, self.computeReward(self.prevState, state),
                   finalStateFeatures)

        # Reset attributes
        self.prevState = None
        self.prevStateFeatures = None
        self.prevAction = None
        self.prevReward = None

        # Decay epsilon throughout training
        epsilon_decay = (self.getNumTraining() - 1) / self.getNumTraining()
        self.setEpsilon(self.epsilon * epsilon_decay)

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)
