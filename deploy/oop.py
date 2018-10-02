from frozenlakegame import frozenlakegame
from network import Network
import matplotlib.pyplot as plt
import numpy as np
import time


if __name__ == "__main__":

    # Load checkpoint
    
    env = frozenlakegame(R=-0.01)  # Instantiate the environment

    PG = Network() # Instantiate the Neural Network
    
    RENDER_ENV = False
    num_episodes = 100000
    max_steps = 40
    wins = 0

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0
        while steps < max_steps:
            if RENDER_ENV:
                env.show()

            # 1. Choose an action based on observation
            action = PG.choose_action(state, steps / max_steps)

            # 2. Take action in the environment
            new_state, reward = env.step(action)

            # 4. Store transition for training
            PG.store_transition(state, action, reward)    

            if env.terminal():
                # 5. Train neural network
                discounted_episode_rewards_norm = PG.learn()
                if reward == 1:
                    wins += 1
                if episode != 0 and episode % 100 == 0:
                    print("{e:d}\t{r:1.4f}".format(e=episode, r=wins/(episode + 1)))
                break
            elif episode % 100 == 0 and steps == max_steps - 1:
                discounted_episode_rewards_norm = PG.learn()
                print("{e:d}\t{r:1.4f}".format(e=episode, r=wins/(episode + 1)))
            steps += 1
            
    print("Episodes: {}, Wins: {}, Winrate: {}".format(episode + 1 , wins, wins/(episode + 1)))