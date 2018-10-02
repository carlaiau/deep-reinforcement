from frozenlakegame import frozenlakegame
import numpy as np
import tensorflow as tf

"""agent.py: Implementation of random action agent for COSC470 Assignment 3.
"""

__author__      = "Lech Szymanski"
__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "lechszym@cs.otago.ac.nz"

# Instantiate the game

env = frozenlakegame(R=-0.05)
# Number of learning episodes
num_episodes = 10100

show = True
normalize_r = False
gamma = 0.95

# Maximum number of steps per episode
max_steps_per_episode = 50

# Hypers
alpha = 1e-4

def discount(r, gamma, normal):
    discount = np.zeros_like(r)
    G = 0.0
    for i in reversed(range(0, len(r))):
        G = G * gamma + r[i]
        discount[i] = G
    # Normalize 
    if normal:
        mean = np.mean(discount)
        std = np.std(discount)
        discount = (discount - mean) / (std)
    return discount


def run_episode():
    
    states, actions, rewards = [], [], []
    state = env.reset()
    episode_num = 0 
    reward = 0
    
    for s in range(max_steps_per_episode):
        if env.terminal():
            break # This environment has been solved

        if show:
            env.show()

        # Run State Through Policy & Calculate Action
        feed = {X: np.reshape(state, (-1, 4, 4, 3)) }
        action = sess.run(action_to_take, feed_dict=feed)[0]
        print(action)
        # Perform Action
        new_state, reward = env.step(action)

        # Store Results
        states.append(state)
        rewards.append(reward)
        actions.append(action)
        
        # Reset reward value for next move
        reward = 0
        state = new_state
        
        
    
    return np.array(states), np.array(actions), np.array(rewards), \
    np.array(discount(rewards, gamma, normalize_r)), s


# Define the Neural network 
X = tf.placeholder(tf.float32, (None,4,4,3), name='X')
Y = tf.placeholder(tf.int32, (None,), name='actions')
R = tf.placeholder(tf.float32, (None,), name='reward')
N = tf.placeholder(tf.float32, (None), name='episodes')
D_R = tf.placeholder(tf.float32, (None,), name='discounted_reward')

# Policy Network
conv1 = tf.layers.conv2d(
    inputs = X,
    filters = 8,
    kernel_size = 4,
    strides = 1,
    padding = 'valid',
    activation = tf.nn.relu,
    name='conv1')

dense = tf.layers.dense(
    inputs = conv1, 
    units = 128, 
    activation = tf.nn.relu,
    name = 'fc')

denser = tf.layers.dense(
    inputs = dense, 
    units = 16, 
    activation = tf.nn.relu,
    name = 'fc2')
    
logits = tf.layers.dense(
    inputs = tf.layers.flatten(denser), 
    units = 4, 
    name='logits')

aprob = tf.nn.softmax(logits)

action_to_take = tf.argmax(input=logits, axis=1)

with tf.name_scope('loss'):
    neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=Y)
    loss = tf.reduce_mean(neg_log_prob * D_R)

with tf.name_scope('train'):
    train_op = tf.train.AdamOptimizer(alpha).minimize(loss)

# Initialize Session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)



step = 0
while step < num_episodes+1:
    # Gather Training Data
    s, a, r, d_r, n = run_episode() # this is called every epoch
    mean_reward = np.sum(r)/n
    print('Epoch: {} Training Episodes: {}  Average Reward: {:4.2f}'.format(step, n, mean_reward))    
    sess.run(train_op, feed_dict={X:s, Y:a, D_R:d_r})
    step += 1