# gym-tic-tac-toe3D
-------------------
OpenAI Gym Environment for a Two-Player 3D Tic-Tac-Toe.

# Requirements
----------------
- gym
- Numpy
- Matplotlib

# Install
----------------------
```python
pip install gym-tic-tac-toe3D
```
# How it Works
Tic Tac Toe is usually played on a 3x3 grid where the objective is for one player to line up their tokens in a straight line of three. This is an extremely easy and trivial game; however, one can extend the difficulty by stacking 3x3 layers to create a 3x3x3 cube. Now the objective is to line up three tokens in any of the directions. 

Here are three example games where the action was randomly generated:

![Game 1](ex1.gif)

![Game 2](ex2.gif)

![Game 3](ex3.gif)

# How to Use

The environment is a Two-Player Game, where Blue denotes Player 1 and Red denotes Player 2. The `state` and `action`  contains all 27 possible positions, 9 for the first layer, 9 for the second, and 9 for the third.

The input has three possible integer values for eac position, -1 for opponent, 0 for empty, and 1 for current player. Note that no matter whom the Player is, these values holds true. The `reward` is a two value list where the first index represents the reward for Player 1 and the second for Player 2. The `reward` value should only be used after the game is completed. Players are rewarded for wins and receive extra points for how fast they win; while players are penalized for losing and how fast they lost. For example, after a game has completed the final reward could be `[20, -10]`, which rewards Player 1 `20` points while penalizing Player 2 `-10` points.

Here is an example on how to create the environment:

```python
import gym
import gym_tic_tac_toe3D
env = gym.make("tic_tac_toe3D-v0")

games = 3  # best of three
player1_reward = 0
player2_reward = 0
for i in range(0, games):
    state = env.reset()
    done = False
    player = 1
    while not done:
        env.render(player=player)
        plt.pause(0.5)
        while True:
            action = env.action_space.sample()
            # Need to check if action is available in state space
            if state[action] == 0:  
                break
    
        state, reward, done, info = env.step(action, player=player)
        # switch players
        if player == 1:
            player = 2
        else:
            player = 1
    # final render after completion of game to see final move
    env.render(player=player)
    plt.pause(1)
    player1_reward += reward[0]
    player2_reward += reward[1]
    
    
```
