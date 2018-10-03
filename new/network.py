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
        learning_rate=0.01,
        reward_decay=0.95,
        load_path=None,
        save_path=None
    ):
        
        self.lr = learning_rate
        self.gamma = reward_decay
        self.episode_state = []
        self.episode_actions = []
        self.episode_rewards = []

        self.save_path = None
        if save_path is not None:
            self.save_path = save_path

        self.build_network()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


        # 'Saver' op to save and restore all the variables
        self.saver = tf.train.Saver()

        # Restore model
        if load_path is not None:
            self.load_path = load_path
            self.saver.restore(self.sess, self.load_path)            
    
    def clear_transitions(self):
        self.episode_state = []
        self.episode_rewards = []
        self.episode_actions = []

    def store_transition(self, s, a, r):
        
        self.episode_state.append(s)
        self.episode_rewards.append(r)
        self.episode_actions.append(a)

    def choose_action(self, state):
        ''' 
        Rather than choosing the action that has the highest value, we are using 
        a biased sample, so that sometimes the action undertaken is not the best 
        performing.  Although this will take longer to converge, it will allow us 
        to explore actions not in our policy. As our policy becomes more confident
        in one specific action, then the choice will naturally become less random
        '''
        state = state[np.newaxis, :]
        actions = self.sess.run(self.outputs_softmax, feed_dict = {self.X: state})
        #if np.random.randint(1, 101) > 80:
        #    return np.random.randint(0, 4)
        #return np.random.choice(range(len(actions.ravel())), p=actions.ravel())
        
        #print(actions)
        #return np.argmax(actions[0])
        return np.random.choice(range(actions.shape[1]), p=actions.ravel())

    def learn(self, milestone=False):
        total_reward = self.discount_and_normalise_rewards()
        #print(self.episode_actions)
        # Train on episode
        self.sess.run(self.train_op, feed_dict={
             self.X: self.episode_state, 
             self.Y: self.episode_actions,
             self.total_reward: total_reward
        })

        # Reset the episode data
        self.episode_state, self.episode_actions, self.episode_rewards  = [], [], []

        # Save checkpoint
        if milestone == True and self.save_path is not None:
            save_path = self.saver.save(self.sess, self.save_path)
        return total_reward

    '''
    Discount and normalise the reward.
    Early on during my development, zero standard deviation results were breaking my model
    so I return zero when one is found 
    '''
    def discount_and_normalise_rewards(self):
        tot = 0
        rewards = [0] * len(self.episode_rewards)
        for t in reversed(range(len(self.episode_rewards))):
            tot = tot * self.gamma + self.episode_rewards[t]
            rewards[t] = tot

        # Standard deviations are equal to zero, return zeros
        # we won't learn anything from this
        return rewards
        
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
            padding = 'same', # this will keep it as a 4 * 4 grid
            activation = tf.nn.relu,
            kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer = tf.constant_initializer(0.1),
            
            name='conv1')   
        
        conv2 = tf.layers.conv2d(
            inputs = conv1,
            filters = 16,
            kernel_size = [2, 2],
            strides = 1,
            padding = 'same', # This will reduce it to a 3 * 3 grid
            activation = tf.nn.relu,
            kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer = tf.constant_initializer(0.1),
            name='conv2')
        
        flat_convs = tf.layers.flatten(
            inputs=conv2
        )
        
        dense = tf.layers.dense(
            inputs = flat_convs, 
            units = 32, 
            activation = tf.nn.tanh,
            kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer = tf.constant_initializer(0.1),
            name = 'fc'
        )    
        self.logits = tf.layers.dense(
            inputs = dense,
            units = 4, 
            kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer = tf.constant_initializer(0.1),
            name='logits')

        # Softmax outputs
        self.outputs_softmax = tf.nn.softmax(logits=self.logits, name='action_probabilities')
        
        with tf.name_scope('loss'): 

            # This is the standard type of loss function, that I couldn't get to perform well
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.Y, logits=self.logits, name="cross_entropy")
            loss = tf.reduce_mean(cross_entropy / - self.total_reward)
            

             
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)    