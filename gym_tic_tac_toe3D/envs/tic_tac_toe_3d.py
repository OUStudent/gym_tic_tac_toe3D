import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

class TicTacToe3D(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.state = []
        super().__init__()
        self.action_space = spaces.Discrete(27)
        self.board = np.asarray([0] * 27).reshape(1, 27)
        self.turn_count = 0
        self.done = False
        self.ax = None
        self.axes = [5, 5, 5]
        self.data = np.ones([3, 3, 3], dtype=np.bool)
        self.colors = np.empty(self.axes + [4], dtype=np.float32)
        self.alpha = 0.2
        self.colors[0] = [1, 1, 1, self.alpha]
        self.colors[1] = [1, 1, 1, self.alpha]
        self.colors[2] = [1, 1, 1, self.alpha]
        self.colors[3] = [1, 1, 1, self.alpha]
        self.colors[4] = [1, 1, 1, self.alpha]
        self.previous_action = None
        self.start = 0
        self.indices_win = None

    def set_up_render(self):

        fig = plt.figure(1)
        self.ax = fig.add_subplot(111, projection='3d')

        def explode(data):
            shape_arr = np.array(data.shape)
            size = shape_arr[:3] * 2 - 1
            exploded = np.zeros(np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
            exploded[::2, ::2, ::2] = data
            return exploded

        self.data = explode(self.data)

        plt.ion()
        self.ax.voxels(self.data, facecolors=self.colors, edgecolors='grey', shade=False)
        self.ax.set_xticks([0, 2, 4])
        self.ax.set_xticklabels(["X0", "X1", "X2"])
        self.ax.set_yticks([0, 2, 4])
        self.ax.set_yticklabels(["Y0", "Y1", "Y2"])
        self.ax.set_zticks([0, 2, 4])
        self.ax.set_zticklabels(["Z0", "Z1", "Z2"])
        plt.suptitle("Tic-Tac-Toe3D")
        plt.show()

    def get_indices(self, action):
        if action < 9:
            i3 = 0
            if action < 3:
                i1 = 0
            elif action < 6:
                i1 = 2
            else:
                i1 = 4
        elif action < 18:
            i3 = 2
            if action < 12:
                i1 = 0
            elif action < 15:
                i1 = 2
            else:
                i1 = 4
        else:
            i3 = 4
            if action < 21:
                i1 = 0
            elif action < 24:
                i1 = 2
            else:
                i1 = 4
        i2 = (action % 3) * 2
        return i1, i2, i3

    def render(self, mode='human', close=False, player=1):

        if self.start == 0:
            self.set_up_render()
            return

        if self.done:
            t = np.where(np.sum(self.colors == np.asarray([1., 0, 0, 1.]), 3) == 4)
            t = [t[0], t[1], t[2]]
            for i in range(0, len(t[0])):
                self.colors[t[0][i], t[1][i], t[2][i]] = [1., 0.5, 0.5, 1.]

            t = np.where(np.sum(self.colors == np.asarray([0, 0, 1, 1.]), 3) == 4)
            t = [t[0], t[1], t[2]]
            for i in range(0, len(t[0])):
                self.colors[t[0][i], t[1][i], t[2][i]] = [0.5, 0.5, 1, 1.]

            if player == 1:
                color = [1, 0, 0, 1]
            else:
                color = [0, 0, 1, 1]
            for index_win in self.indices_win:
                i1, i2, i3 = self.get_indices(index_win)
                self.colors[i1, i2, i3] = color
        else:
            i1, i2, i3 = self.get_indices(self.previous_action)
            if player == 1:
                self.colors[i1, i2, i3] = [1, 0, 0, 1]
            else:
                self.colors[i1, i2, i3] = [0, 0, 1, 1]
        self.ax.voxels(self.data, facecolors=self.colors, edgecolors='grey', shade=False)


    def step(self, action, player=1):
        self.start = 1
        if self.done:
            print("Game Already Over")
            return None

        if self.board[0][action] != 0:
            print("Position Already Taken")
            return self.board[0], [0, 0], self.done, {"draw": False,
                                                   "winning_player": -1,
                                                   "turn_count": self.turn_count}

        viable_moves = np.where(self.board[0] == 0)[0].tolist()
        if len(viable_moves) == 0:
            self.done = True
            return self.board[0], [0, 0], self.done, {"draw": True,
                                                   "winning_player": -1,
                                                   "turn_count": self.turn_count}
        if player == 1:
            self.board[0][action] = 1
        else:
            self.board[0][action] = -1

        self.previous_action = action
        win = self.check_win(player=player)
        if win:
            self.done = True
            if self.turn_count <= 3:
                reward = [20, -10]
            elif self.turn_count <= 5:
                reward = [18, -8]
            elif self.turn_count <= 7:
                reward = [16, -6]
            elif self.turn_count <= 9:
                reward = [14, -4]
            else:
                reward = [10, -2]

            if player == 2:
                reward = reward[::-1]

            return self.board[0], reward, self.done, {"draw": False,
                                                   "winning_player": player,
                                                   "turn_count": self.turn_count}
        else:
            if player == 2:
                self.turn_count += 1
            temp = self.board
            if player == 1:
                temp = np.negative(temp)
            return temp[0], [0, 0], self.done, {"draw": False,
                                             "winning_player": -1,
                                             "turn_count": self.turn_count}

    def reset(self):
        self.state = []
        self.board = np.asarray([0] * 27).reshape(1, 27)
        self.turn_count = 0
        self.done = False
        self.ax = None
        self.axes = [5, 5, 5]
        self.data = np.ones([3, 3, 3], dtype=np.bool)
        self.colors = np.empty(self.axes + [4], dtype=np.float32)
        self.alpha = 0.2
        self.colors[0] = [1, 1, 1, self.alpha]
        self.colors[1] = [1, 1, 1, self.alpha]
        self.colors[2] = [1, 1, 1, self.alpha]
        self.colors[3] = [1, 1, 1, self.alpha]
        self.colors[4] = [1, 1, 1, self.alpha]
        self.previous_action = None
        self.start = 0
        self.indices_win = None
        return self.board[0]

    def check_win(self, player=1):
        winning_states_layers = [[1, 1, 1, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 1, 1, 1, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 1, 1, 1],
                                 [1, 0, 0, 1, 0, 0, 1, 0, 0],
                                 [0, 1, 0, 0, 1, 0, 0, 1, 0],
                                 [0, 0, 1, 0, 0, 1, 0, 0, 1],
                                 [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                 [0, 0, 1, 0, 1, 0, 1, 0, 0]]
        if player == 2:
            layers = np.split(np.negative(self.board[0]), 3)
        else:
            layers = np.split(self.board[0], 3)
        layer_count = 0
        for layer in layers:
            row = 0
            t = layer_count*9
            for winning_layer in winning_states_layers:
                temp = layer * winning_layer
                if np.sum(temp == winning_layer) == len(temp):
                    if row == 0:
                        self.indices_win = [t+0, t+1, t+2]
                    elif row == 1:
                        self.indices_win = [t+3, t+4, t+5]
                    elif row == 2:
                        self.indices_win = [t+6, t+7, t+8]
                    elif row == 3:
                        self.indices_win = [t+0, t+3, t+6]
                    elif row == 4:
                        self.indices_win = [t+1, t+4, t+7]
                    elif row == 5:
                        self.indices_win = [t+2, t+5, t+8]
                    elif row == 6:
                        self.indices_win = [t+0, t+4, t+8]
                    elif row == 7:
                        self.indices_win = [t+2, t+4, t+6]
                    return True
                row += 1
            layer_count += 1
        for i in range(0, 9):
            if layers[0][i] == layers[1][i] == layers[2][i] == 1:
                self.indices_win = [i, 9+i, 18+i]
                return True
        if layers[0][0] == layers[1][3] == layers[2][6] == 1:
            self.indices_win = [0, 12, 24]
            return True
        elif layers[0][1] == layers[1][4] == layers[2][7] == 1:
            self.indices_win = [1, 13, 25]
            return True
        elif layers[0][2] == layers[1][5] == layers[2][8] == 1:
            self.indices_win = [2, 14, 26]
            return True
        elif layers[0][6] == layers[1][3] == layers[2][0] == 1:
            self.indices_win = [6 , 12 , 18]
            return True
        elif layers[0][7] == layers[1][4] == layers[2][1] == 1:
            self.indices_win = [7 , 13 , 19]
            return True
        elif layers[0][8] == layers[1][5] == layers[2][2] == 1:
            self.indices_win = [8 , 14, 20]
            return True
        elif layers[0][0] == layers[1][1] == layers[2][2] == 1:
            self.indices_win = [0 ,10 , 20]
            return True
        elif layers[0][3] == layers[1][4] == layers[2][5] == 1:
            self.indices_win = [3 , 13 , 23]
            return True
        elif layers[0][6] == layers[1][7] == layers[2][8] == 1:
            self.indices_win = [6 , 16 , 26]
            return True
        elif layers[0][2] == layers[1][1] == layers[2][0] == 1:
            self.indices_win = [2 , 10 , 18]
            return True
        elif layers[0][5] == layers[1][4] == layers[2][3] == 1:
            self.indices_win = [5 , 13 , 21]
            return True
        elif layers[0][8] == layers[1][7] == layers[2][6] == 1:
            self.indices_win = [8 ,16 , 24]
            return True
        elif layers[0][0] == layers[1][4] == layers[2][8] == 1:
            self.indices_win = [0 , 13, 26]
            return True
        elif layers[0][2] == layers[1][4] == layers[2][6] == 1:
            self.indices_win = [2 , 13 , 24]
            return True
        elif layers[0][6] == layers[1][4] == layers[2][2] == 1:
            self.indices_win = [6, 13, 20]
            return True
        elif layers[0][8] == layers[1][4] == layers[2][0] == 1:
            self.indices_win = [8 ,13 ,18]
            return True
        else:
            return False


