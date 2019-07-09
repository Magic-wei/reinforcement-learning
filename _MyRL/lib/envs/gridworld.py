import io
import numpy as np
import sys
from gym.envs.toy_text import discrete
from enum import Enum, unique

@unique
class Action(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    UPRIGHT = 4
    DOWNRIGHT = 5
    DOWNLEFT = 6
    UPLEFT = 7

UP = Action.UP.value
RIGHT = Action.RIGHT.value
DOWN = Action.DOWN.value
LEFT = Action.LEFT.value
UPRIGHT = Action.UPRIGHT.value
DOWNRIGHT = Action.DOWNRIGHT.value
DOWNLEFT = Action.DOWNLEFT.value
UPLEFT = Action.UPLEFT.value
    
class GridworldEnv(discrete.DiscreteEnv):
    """
    Grid World environment from Sutton's Reinforcement Learning book chapter 4.
    You are an agent on an MxN grid and your goal is to reach the terminal
    state at the top left or the bottom right corner.

    For example, a 4x4 grid looks as follows:

    T  o  o  o
    o  x  o  o
    o  o  o  o
    o  o  o  T

    x is your position and T are the two terminal states.

    You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave you in your current state.
    You receive a reward of -1 at each step until you reach a terminal state.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape=[4,4], actionmode=4):
        """
        Args:
            shape: Grid size.
            actionmode: this example only support 4 (default) or 8 actions.
        """
        
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')

        self.shape = shape

        nS = np.prod(shape) # np.prod 计算所有元素的乘积
        nA = actionmode

        MAX_Y = shape[0]
        MAX_X = shape[1]

        P = {}
        grid = np.arange(nS).reshape(shape)
        it = np.nditer(grid, flags=['multi_index']) # numpy 迭代器，multi_index指多维迭代，对于gridworld问题，nS表示状态总数，但这些状态要分布到二维栅格内，还需要有坐标对应。这个用法方便迭代。

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            # P[s][a] = (prob, next_state, reward, is_done)
            P[s] = {a : [] for a in range(nA)} # init

            # terminal flag function
            is_done = lambda s: s == 0 or s == (nS - 1)
            
            # rewards
            reward = 0.0 if is_done(s) else -1.0

            # We're stuck in a terminal state
            if is_done(s):
#                 P[s][UP] = [(1.0, s, reward, True)]
#                 P[s][RIGHT] = [(1.0, s, reward, True)]
#                 P[s][DOWN] = [(1.0, s, reward, True)]
#                 P[s][LEFT] = [(1.0, s, reward, True)]
                for a in range(nA):
                    P[s][a] = [(1.0, s, reward, True)]
            # Not a terminal state
            else:
                next_state = list(np.zeros(8))
                next_state[UP] = s if y == 0 else s - MAX_X
                next_state[RIGHT] = s if x == (MAX_X - 1) else s + 1
                next_state[DOWN] = s if y == (MAX_Y - 1) else s + MAX_X
                next_state[LEFT] = s if x == 0 else s - 1
                next_state[UPRIGHT] = s if y == 0 or x == (MAX_X - 1) else  s - MAX_X + 1
                next_state[DOWNRIGHT] = s if y == (MAX_Y - 1) or x == (MAX_X - 1) else  s + MAX_X + 1
                next_state[DOWNLEFT] = s if y == (MAX_Y - 1) or x == 0 else s + MAX_X - 1
                next_state[UPLEFT] = s if y == 0 or x == 0 else s - MAX_X - 1
                for a in range(nA):
                    P[s][a] = [(1.0, next_state[a], reward, is_done(next_state[a]))]
#                 P[s][UP] = [(1.0, ns_up, reward, is_done(ns_up))]
#                 P[s][RIGHT] = [(1.0, ns_right, reward, is_done(ns_right))]
#                 P[s][DOWN] = [(1.0, ns_down, reward, is_done(ns_down))]
#                 P[s][LEFT] = [(1.0, ns_left, reward, is_done(ns_left))]
    
            it.iternext()

        # Initial state distribution is uniform
        isd = np.ones(nS) / nS

        # We expose the model of the environment for educational purposes
        # This should not be used in any model-free learning algorithm
        self.P = P

        super(GridworldEnv, self).__init__(nS, nA, P, isd) # Python 2 version | In Python 3, can also be written as super().__init__(nS, nA, P, isd)

    def _render(self, mode='human', close=False):
        """ Renders the current gridworld layout

         For example, a 4x4 grid with the mode="human" looks like:
            T  o  o  o
            o  x  o  o
            o  o  o  o
            o  o  o  T
        where x is your position and T are the two terminal states.
        """
        if close:
            return

        outfile = io.StringIO() if mode == 'ansi' else sys.stdout

        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            if self.s == s:
                output = " x "
            elif s == 0 or s == self.nS - 1:
                output = " T "
            else:
                output = " o "

            if x == 0:
                output = output.lstrip()
            if x == self.shape[1] - 1:
                output = output.rstrip()

            outfile.write(output)

            if x == self.shape[1] - 1:
                outfile.write("\n")

            it.iternext()
