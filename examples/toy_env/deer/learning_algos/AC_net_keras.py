"""
Code for the actor-critic "DDPG" (https://arxiv.org/abs/1509.02971)

"""

import numpy as np
from ..base_classes import LearningAlgo as ACNetwork
from .NN_keras import NN # Default Neural network used
from tensorflow.keras.optimizers import SGD,RMSprop
from tensorflow.keras import backend as K

try:
    import tensorflow as tf
    assert(K.backend()=="tensorflow")
except:
    print('Error : Currently only Tensorflow is supported as a backend for AC_net_keras. You can make the switch in the file ~/.keras/keras.json')

class MyACNetwork(ACNetwork):
    """
    Actor-critic learning (using Keras) with Deep Deterministic Policy Gradient (DDPG) for the continuous action domain
    
    Parameters
    -----------
    environment : object from class Environment
        The environment in which the agent evolves.
    rho : float
        Parameter for rmsprop. Default : 0.9
    rms_epsilon : float
        Parameter for rmsprop. Default : 0.0001
    momentum : float
        Momentum for SGD. Default : 0
    clip_norm : float
        The gradient tensor will be clipped to a maximum L2 norm given by this value.
    freeze_interval : int
        Period during which the target network is freezed and after which the target network is updated. Default : 1000
    batch_size : int
        Number of tuples taken into account for each iteration of gradient descent. Default : 32
    update_rule: str
        {sgd,rmsprop}. Default : rmsprop
    random_state : numpy random number generator
        Set the random seed.
    double_Q : bool, optional
        Activate or not the double_Q learning.
        More informations in : Hado van Hasselt et al. (2015) - Deep Reinforcement Learning with Double Q-learning.
    neural_network_critic : object, optional
        default is deer.learning_algos.NN_keras
    neural_network_actor : object, optional
        default is deer.learning_algos.NN_keras
    """

    def __init__(self, environment, rho=0.9, rms_epsilon=0.0001, momentum=0, clip_norm=0, freeze_interval=1000, batch_size=32, update_rule="rmsprop", random_state=np.random.RandomState(), double_Q=False, neural_network_critic=NN, neural_network_actor=NN):
        """ Initialize environment
        
        """
        ACNetwork.__init__(self,environment, batch_size)

        self._rho = rho
        self._rms_epsilon = rms_epsilon
        self._momentum = momentum
        self._clip_norm = clip_norm
        self._freeze_interval = freeze_interval
        self._double_Q = double_Q
        self._random_state = random_state
        self._nActions=environment.nActions()
        self.update_counter = 0
        
        # self.sess = tf.Session()
        # K.set_session(self.sess)
        
        Q_net = neural_network_critic(self._batch_size, self._input_dimensions, self._n_actions, self._random_state, True)
        
        self.q_vals, self.params, self.inputsQ = Q_net._buildDQN()
        
        if (update_rule=="sgd"):
            optimizer = SGD(lr=self._lr, momentum=self._momentum, nesterov=False, clipnorm=self._clip_norm)
        elif (update_rule=="rmsprop"):
            optimizer = RMSprop(lr=self._lr, rho=self._rho, epsilon=self._rms_epsilon, clipnorm=self._clip_norm)
        else:
            raise Exception('The update_rule '+update_rule+ 'is not implemented.')
        
        self.q_vals.compile(optimizer=optimizer, loss='mse')
       
        self.next_q_vals, self.next_params, self.next_inputsQ = Q_net._buildDQN()
        self.next_q_vals.compile(optimizer='rmsprop', loss='mse') #The parameters do not matter since training is done on self.q_vals

        self._resetQHat()
        

        policy_net = neural_network_actor(self._batch_size, self._input_dimensions, self._n_actions, self._random_state, False)
        self.policy, self.params_policy = policy_net._buildDQN()
        self.policy.compile(optimizer=optimizer, loss='mse')
        self.next_policy, self.next_params_policy = policy_net._buildDQN()
        self.next_policy.compile(optimizer=optimizer, loss='mse')
        
        
        
        ### self.policy
        #self.action_grads = tf.gradients(self.q_vals.output,self.inputsQ[-1])  #GRADIENTS for policy update
       
        
        #self.sess.run(tf.initialize_all_variables())


    def getAllParams(self):
        """ Get all parameters used by the learning algorithm

        Returns
        -------
        Values of the parameters: list of numpy arrays
        """
        params_value=[]
        for i,p in enumerate(self.params):
            params_value.append(K.get_value(p))
        for i,p in enumerate(self.params_policy):
            params_value.append(K.get_value(p))
        
        return params_value

    def setAllParams(self, list_of_values):
        """ Set all parameters used by the learning algorithm

        Arguments
        ---------
        list_of_values : list of numpy arrays
             list of the parameters to be set (same order than given by getAllParams()).
        """
        for i,p in enumerate(self.params):
            K.set_value(p,list_of_values[i])
        for j,p in enumerate(self.params_policy):
            K.set_value(p,list_of_values[j+i+1])

    def train(self, states_val, actions_val, rewards_val, next_states_val, terminals_val):
        """
        Train the actor-critic algorithm from one batch of data.

        Parameters
        -----------
        states_val : numpy array of objects
            Each object is a numpy array that relates to one of the observations
            with size [batch_size * history size * size of punctual observation (which is 2D,1D or scalar)]).
        actions_val : numpy array of objects with size [self._batch_size].
            Each object is a numpy array of floats with size [len(self._nActions)]
            actions[i] is the action taken after having observed states[:][i].
        rewards_val : numpy array of floats with size [self._batch_size]
            rewards[i] is the reward obtained for taking actions[i-1].
        next_states_val : numpy array of objects
            Each object is a numpy array that relates to one of the observations
            with size [batch_size * history size * size of punctual observation (which is 2D,1D or scalar)]).
        terminals_val : numpy array of booleans with size [self._batch_size]
            terminals[i] is True if the transition leads to a terminal state and False otherwise


        Returns
        -------
        Average loss of the batch training
        Individual losses for each tuple
        """
        if self.update_counter % self._freeze_interval == 0:
            self._resetQHat()
        

        ### Tain self.q_vals
        next_actions_val=self.next_policy.predict(next_states_val.tolist())

        ns_list=next_states_val.tolist()
        ns_list.append( next_actions_val )
        next_q_vals = self.next_q_vals.predict(  ns_list  )
        
        not_terminals=np.invert(terminals_val).astype(float)
        
        target = rewards_val + not_terminals * self._df * next_q_vals.reshape((-1))
        
        s_list=states_val.tolist()
        s_list.append( np.array(actions_val.tolist())  )
        
        # In order to obtain the individual losses, we predict the current Q_vals and calculate the diff
        q_vals=self.q_vals.predict( s_list ).reshape((-1))
        diff_q = - q_vals + target 
        loss_ind_q=pow(diff_q,2)
        
        loss_q=self.q_vals.train_on_batch( s_list , target ) 
        
        
        ### Train self.policy
        cur_action=self.policy.predict(states_val.tolist())
        cur_action=self.clip_action(cur_action)
        gg=self.gradients(states_val.tolist(),cur_action)
        
        target_action=self.clip_action(cur_action+gg)
        
        # Calculation of the individual losses for the policy network
        diff_policy = - cur_action + target_action
        loss_ind_policy=np.sum(pow(diff_policy,2),axis=-1)

        loss_policy=self.policy.train_on_batch(states_val.tolist(), target_action)
                        
        self.update_counter += 1        
        
        
        return loss_q+loss_policy,loss_ind_q+loss_ind_policy


    def clip_action(self, action):
        """
        Clip the possible actions if it is outside the action space defined by self._nActions
        self._nActions is given as [[low_action1,high_action1],[low_action2,high_action2], ...]
        """
        return np.clip(action,np.array(self._nActions)[:,0],np.array(self._nActions)[:,1])
    

    def gradients(self, states, actions):
        """
        Returns the gradients on the Q-network for the different actions (used for policy update)
        """
        # combine state features with action
        input_list = states.copy()
        input_list.append(actions)

        # inputs need to be tf.Variable to calculate gradients
        input_list = [tf.Variable(input, dtype=tf.float32) for input in input_list]

        with tf.GradientTape() as tape:
            q_vals = self.q_vals(input_list)

        grads = tape.gradient(q_vals, input_list)

        #last entry in grads corresponds to the gradients of the q_vals with respect to the action
        out = grads[-1].numpy()

        return out

    def chooseBestAction(self, state, *args, **kwargs):
        """ Get the best action for a pseudo-state

        Arguments
        ---------
        state : one pseudo-state

        Returns
        -------
        best_action : float
        estim_value : float
        """        
        
        best_action=self.policy.predict([np.expand_dims(s,axis=0) for s in state])
        best_action=self.clip_action(best_action)
        
        the_list=[np.expand_dims(s,axis=0) for s in state]
        the_list.append( best_action )
        estim_value=(self.q_vals.predict(the_list)[0,0])
        
        return best_action[0],estim_value
        
    def _resetQHat(self):
        for i,(param,next_param) in enumerate(zip(self.params, self.next_params)):
            K.set_value(next_param,K.get_value(param))
