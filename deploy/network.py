"""
Policy Gradient Reinforcement Learning
Uses a 3 layer neural network as the policy network
Uses tf.layers to build the neural network

"""
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops

class Network:

    def __init__(
        self,
        learning_rate=0.001,
        reward_decay=0.95
    ):
        
        self.lr = learning_rate
        self.gamma = reward_decay
        self.episode_state = []
        self.episode_actions = []
        self.episode_rewards = []
        self.build_network()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def store_transition(self, s, a, r):
        
        self.episode_state.append(s)
        self.episode_rewards.append(r)
        self.episode_actions.append(a)

    def choose_action(self, state, progression):
        ''' 
        Rather than choosing the action that has the highest value, we are using 
        a biased sample, so that sometimes the action undertaken is not the best 
        performing.  Although this will take longer to converge, it will allow us 
        to explore actions not in our policy. As our policy becomes more confident
        in one specific action, then the choice will naturally become less random
        '''
        state = state[np.newaxis, :]
        actions = self.sess.run(self.outputs_softmax, feed_dict = {self.X: state})
        return np.random.choice(range(len(actions.ravel())), p=actions.ravel())


    def learn(self):
        total_reward = self.discount_and_normalise_rewards()
        
        # Train on episode
        self.sess.run(self.train_op, feed_dict={
             self.X: np.array(self.episode_state), # shape [ examples, number of inputs]
             self.Y: np.array(self.episode_actions), # shape [actions, ]
             self.total_reward: total_reward
        })

        # Reset the episode data
        self.episode_state, self.episode_actions, self.episode_rewards  = [], [], []

        return total_reward

    '''
    Discount and normalise the reward.
    Early on during my development, zero standard deviation results were breaking my model
    so I return zero when one is found 
    '''
    def discount_and_normalise_rewards(self):
        rewards = np.zeros_like(self.episode_rewards)
        tot = 0
        
        for t in reversed(range(len(self.episode_rewards))):
            tot *= self.gamma
            tot += self.episode_rewards[t]
            rewards[t] = tot

        # Standard deviations are equal to zero, return zeros
        # we won't learn anything from this
        if np.std(rewards) == 0:
            return np.zeros_like(self.episode_rewards)
        
        # Normalise by discounting the mean
        rewards -= np.mean(rewards)
        # Divide all rewards by standard deviation
        return rewards / np.std(rewards)

    
    def build_network(self):
        with tf.name_scope('inputs'):
            self.X = tf.placeholder(tf.float32, (None,4,4,3), name="X")
            self.Y = tf.placeholder(tf.int32, [None, ], name="Y")
            self.total_reward = tf.placeholder(tf.float32, [None, ], name="reward")

        conv1 = tf.layers.conv2d(
            inputs = self.X,
            filters = 16,
            kernel_size = [2, 2],
            strides = 1,
            padding = 'valid',
            activation = tf.nn.relu,
            name='conv1')   
        
        conv2 = tf.layers.conv2d(
            inputs = conv1,
            filters = 16,
            kernel_size = [2, 2],
            strides = 1,
            padding = 'valid',
            activation = tf.nn.relu,
            name='conv2')
        
        conv_flatten = tf.layers.flatten(
            inputs = conv2
        )
        dense = tf.layers.dense(
            inputs = conv_flatten, 
            units = 32, 
            activation = tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
            name = 'fc2'
        )
        logits = tf.layers.dense(
            inputs = dense,
            units = 4, 
            name='logits')
        
        # Softmax outputs
        self.outputs_softmax = tf.nn.softmax(logits, name='A3')

        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.Y)
            loss = tf.reduce_mean( -1 * neg_log_prob * self.total_reward) 

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
