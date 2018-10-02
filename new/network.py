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

    def choose_action(self, state):
        ''' 
        Rather than choosing the action that has the highest value, we are using 
        a biased sample, so that sometimes the action undertaken is not the best 
        performing.  Although this will take longer to converge, it will allow us 
        to explore actions not in our policy. As our policy becomes more confident
        in one specific action, then the choice will naturally become less random
        '''
        state = state[np.newaxis, :]
        action_to_choose = self.sess.run(self.action_to_choose, feed_dict = {self.X: state})
        return action_to_choose

        



    def learn(self):
        total_reward = self.discount_and_normalise_rewards()
        

        # Train on episode

        '''

        HAVING a look!
        
        indices = self.sess.run(self.indices, feed_dict={
             self.X: self.episode_state, 
             self.Y: self.episode_actions,
             self.total_reward: total_reward
        })
        print("action indices")
        print(indices)
        print("")
        action_sample = self.sess.run(self.action_sample, feed_dict={
             self.X: self.episode_state, 
             self.Y: self.episode_actions,
             self.total_reward: total_reward
        })
        print("action sample")
        print(action_sample)
        print("")
        action_to_choose = self.sess.run(self.action_to_choose, feed_dict={
             self.X: self.episode_state, 
             self.Y: self.episode_actions,
             self.total_reward: total_reward
        })
        print("action to choose")
        print(action_to_choose)
        print("")
        
        action_prob = self.sess.run(self.act_prob, feed_dict={
             self.X: self.episode_state, 
             self.Y: self.episode_actions,
             self.total_reward: total_reward
        })
        print("action prob")
        print(action_prob)
        print("\n\n")        
        #'''
        self.sess.run(self.train_op, feed_dict={
             self.X: self.episode_state, 
             self.Y: self.episode_actions,
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
        tot = 0
        rewards = [0] * len(self.episode_rewards)
        for t in reversed(range(len(self.episode_rewards))):
            tot = tot * self.gamma + self.episode_rewards[t]
            rewards[t] = tot

        # Standard deviations are equal to zero, return zeros
        # we won't learn anything from this
        if np.std(rewards) == 0:
            return [0] * len(self.episode_rewards)
        
        # Normalise by discounting the mean
        norm_mean, norm_variance = tf.nn.moments(tf.constant(rewards, dtype=tf.float32), axes=None)
        rewards -= norm_mean
        rewards /= tf.sqrt(norm_variance)
        # Divide all rewards by standard deviation

        return self.sess.run(rewards)

    
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
            name='conv1')   
        
        conv2 = tf.layers.conv2d(
            inputs = conv1,
            filters = 16,
            kernel_size = [2, 2],
            strides = 1,
            padding = 'valid', # This will reduce it to a 3 * 3 grid
            activation = tf.nn.relu,
            name='conv2')
        
        flat_convs = tf.layers.flatten(
            inputs=conv2
        )
        
        dense = tf.layers.dense(
            inputs = flat_convs, 
            units = 32, 
            activation = tf.nn.relu,
            name = 'fc'
        )    
        logits = tf.layers.dense(
            inputs = dense,
            units = 4, 
            name='logits')

        # Softmax outputs
        outputs_softmax = tf.nn.softmax(logits=logits, name='action_probabilities')
        
        # Action to choose 
        self.action_sample = tf.multinomial(tf.log(outputs_softmax), 1)
        self.action_to_choose = tf.cast(self.action_sample[0][0], tf.int32)

        with tf.name_scope('loss'):
            '''
            self.indices = tf.range(0, tf.shape(outputs_softmax)[0]) * tf.shape(outputs_softmax)[1] + self.Y
            self.act_prob = tf.gather(tf.reshape(outputs_softmax, [-1]), self.indices)
        
            # surrogate loss
            loss = tf.reduce_sum(tf.multiply(self.act_prob, self.total_reward))
            '''
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.Y)
            loss = tf.reduce_sum(tf.multiply(neg_log_prob, self.total_reward))

        with tf.name_scope('train'):
            self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(loss)    