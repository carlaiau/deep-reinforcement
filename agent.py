from frozenlakegame import frozenlakegame
import numpy as np

"""agent.py: Implementation of random action agent for COSC470 Assignment 3.
"""

__author__      = "Lech Szymanski"
__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "lechszym@cs.otago.ac.nz"

# Instantiate the game

# You can change the reward value of the ice squares - by default it's 0, but it
# might be a good idea to give it a small negative reward if you want your agent
# to pick shorter paths
env = frozenlakegame(R=0)

# Number of learning episodes
num_episodes = 20
# Maximum number of steps per episode
max_steps_per_episode = 40

win_history = []

# Change this to False to skip visualisations during training - they slow everything down
show_vis = True


for e in range(num_episodes):

    # Reset the environment and get the starting state
    #
    # The state is an env.H x env.W colour image with
    # green pixel corresponding to slippery ice,
    # blue pixel corresponding to water (R=-1)
    # red pixel corresponding to goal (R=1)
    # white pixel indicating the player's location
    s = env.reset()

    # Visualisation of the current state
    if show_vis:
        env.show(s=s)

    for i in range(max_steps_per_episode):

        # If the environment got to the terminal state
        # start a new episode
        if env.terminal():
            break

        # Pick a random action - this is where your policy
        # would make some choices of actions - there are
        # 4 actions index from 0 to 3 corresponding to
        # player's movement towards N,E,S and W.
        #
        # ...
        a = np.random.randint(0,env.num_actions)

        # Execute the action, get the next state and the reward
        s, R = env.step(a)

        # Here you need to do something where you use the reward
        # to update your policy in online mode, or or store the state reward and
        # action for later update to the policy if you're training
        # the policy in the batch mode
        #
        # ...

        # Shows the new state of the environment - only the player's location
        # might change
        if show_vis:
            env.show(s=s)

    # If you're training in batch mode, here's where you might want to
    # use the stored buffer of episode states,actions and rewards in order
    # to improve the policy
    #
    # ...

    infoStr = "Episode %d, " % (e+1)
    if R==1:
        infoStr += "win, "
        win_history.append(1)
    else:
        win_history.append(0)
        if R==-1:
            infoStr += "loss, "
        else:
            infoStr += "timeout, "

    infoStr += "wins rate: %.2f" % (np.sum(win_history)/len(win_history))


    print(infoStr)

# Show the final score (ratio of wins over episodes)
env.show(blocking=True)
