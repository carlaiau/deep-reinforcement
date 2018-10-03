from frozenlakegame import frozenlakegame
from network import Network
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

if __name__ == "__main__":

    
    # Load checkpoint
    load_version = 0
    save_version = load_version + 1
    save_version = 0
    load_path = "output/carlaiau-{}.ckpt".format(load_version)
    if load_version == 0:
        load_path = None
    save_path = "output/carlaiau-{}.ckpt".format(save_version)
    
    
    env = frozenlakegame(R=-0.02)  # Instantiate the environment

    PG = Network(
        learning_rate=0.0001,
        reward_decay=0.99,
        load_path=load_path,
        save_path=save_path
    ) # Instantiate the Neural Network
    
    RENDER_ENV = False
    num_episodes = 50000
    max_steps = 40
    wins = 0
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0
        while steps < max_steps:
            if RENDER_ENV:
                env.show()
          
            action = PG.choose_action(state)
            
            new_state, reward = env.step(action)
            PG.store_transition(state, action, reward)    
            if env.terminal():  
                if reward == 1:
                    wins += 1
                if episode % 500 == 0: # Print and save
                    discounted_episode_rewards_norm = PG.learn(milestone=True)
                    print("{e:d}\t{r:1.4f}".format(e=episode, r=wins/(episode + 1)))
                else:
                    discounted_episode_rewards_norm = PG.learn()
                break

            elif steps == max_steps - 1: # This has reached the end of the game state with no terminal
                
                if episode % 500 == 0: # Print and save
                    discounted_episode_rewards_norm = PG.learn(milestone=True)   
                    print("{e:d}\t{r:1.4f}".format(e=episode, r=wins/(episode + 1)))
                else:
                    discounted_episode_rewards_norm = PG.learn()
            steps += 1