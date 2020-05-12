
import abc


from random import randint, random

import numpy as np

from lspi.sample import Sample

from lspi.domains import Domain
import quadcopter



class QuadcopterDomain(Domain):

    """Chain domain from LSPI paper.

    Very simple MDP. Used to test LSPI methods and demonstrate the interface.
    The state space is a series of discrete nodes in a chain. There are two
    actions: Left and Right. These actions fail with a configurable
    probability. When the action fails to performs the opposite action. In
    otherwords if left is the action applied, but it fails, then the agent will
    actually move right (assuming it is not in the right most state).

    The default reward for any action in a state is 0. There are 2 special
    states that will give a +1 reward for entering. The two special states can
    be configured to appear at the end of the chain, in the middle, or
    in the middle of each half of the state space.

    Parameters
    ----------
    num_states: int
        Number of states in the chain. Must be at least 4.
        Defaults to 10 states.
    reward_location: QuadcopterDomain.RewardLoction
        Location of the states with +1 rewards
    failure_probability: float
        The probability that the applied action will fail. Must be in range
        [0, 1]

    """

    class RewardLocation(object):

        """Location of states giving +1 reward in the chain.

        Ends:
            Rewards will be given at the ends of the chain.
        Middle:
            Rewards will be given at the middle two states of the chain.
        HalfMiddles:
            Rewards will be given at the middle two states of each half
            of the chain.

        """

        Ends, Middle, HalfMiddles = range(3)

    __action_names = ['left', 'right']

    def __init__(self, num_states=10,
                 reward_location=RewardLocation.Ends,
                 failure_probability=.1):
        """Initialize QuadcopterDomain."""
        #if num_states < 4:
        #    raise ValueError('num_states must be >= 4')
        #if failure_probability < 0 or failure_probability > 1:
        #    raise ValueError('failure_probability must be in range [0, 1]')

        self.num_states = int(num_states)
        #self.reward_location = reward_location
        self.failure_probability = failure_probability

        self._state = QuadcopterDomain.__init_random_state(num_states)



        self.key = 'q1'
        QUADCOPTER={self.key: {'position':[1,0,4],'orientation':[0,0,0],'L':0.3,'r':0.1,'prop_size':[10,4.5],'weight':1.2}}

        self.quad_dict = QUADCOPTER

        self.quad = quadcopter.Quadcopter(QUADCOPTER)
        
        self.speed_low = 4000
        self.speed_high = 9000
        self.num_of_possible_speeds = 5
        self.possible_motor_speeds = np.linspace(self.speed_low, self.speed_high, num=self.num_of_possible_speeds).astype(int)

        self.target_position = [1, 0, 0]
        self.target_lin_vel = [0, 0, 0]



    def num_actions(self):
        """Return number of actions.

        Chain domain has 2 actions.

        Returns
        -------
        int
            Number of actions

        """
        return self.num_of_possible_speeds ** self.quad.num_propellers #2

    def current_state(self):
        """Return the current state of the domain.

        Returns
        -------
        numpy.array
            The current state as a 1D numpy vector of type int.

        """
        return self._state

    def apply_action(self, action):
        """Apply the action to the chain.

        If left is applied then the occupied state index will decrease by 1.
        Unless the agent is already at 0, in which case the state will not
        change.

        If right is applied then the occupied state index will increase by 1.
        Unless the agent is already at num_states-1, in which case the state
        will not change.

        The reward function is determined by the reward location specified when
        constructing the domain.

        If failure_probability is > 0 then there is the chance for the left
        and right actions to fail. If the left action fails then the agent
        will move right. Similarly if the right action fails then the agent
        will move left.

        Parameters
        ----------
        action: int
            Action index. Must be in range [0, num_actions())

        Returns
        -------
        sample.Sample
            The sample for the applied action.

        Raises
        ------
        ValueError
            If the action index is outside of the range [0, num_actions())

        """
    

        '''
        action_failed = False
        if random() < self.failure_probability:
            action_failed = True

        # this assumes that the state has one and only one occupied location
        if (action == 0 and not action_failed) \
                or (action == 1 and action_failed):
            new_location = max(0, self._state[0]-1)
        else:
            new_location = min(self.num_states-1, self._state[0]+1)

        next_state = np.array([new_location])
        '''

        motor_speeds = self.get_motor_speed_combination(action)
        print(motor_speeds, 'speed')
        self.quad.set_motor_speeds(self.key, motor_speeds)
        self.quad.update()

        new_position = self.quad.get_position(self.key)
        linear_v = self.quad.get_linear_rate(self.key)

        pos_err = np.subtract(self.target_position, new_position)
        lin_v_err = np.subtract(self.target_lin_vel, linear_v)

        next_state = np.hstack((pos_err, lin_v_err))

        reward = self.calc_reward(new_position, linear_v, self.quad.get_orientation(self.key))
        sample = Sample(self._state.copy(), action, reward, next_state.copy())

        self._state = next_state



        '''
        reward = 0
        if self.reward_location == QuadcopterDomain.RewardLocation.Ends:
            if new_location == 0 or new_location == self.num_states-1:
                reward = 1
        elif self.reward_location == QuadcopterDomain.RewardLocation.Middle:
            if new_location == int(self.num_states/2) \
                    or new_location == int(self.num_states/2 + 1):
                reward = 1
        else:  # HalfMiddles case
            if new_location == int(self.num_states/4) \
                    or new_location == int(3*self.num_states/4):
                reward = 1
        

        sample = Sample(self._state.copy(), action, reward, next_state.copy())
        '''

        return sample

    def reset(self, initial_state=None):
        """Reset the domain to initial state or specified state.

        If the state is unspecified then it will generate a random state, just
        like when constructing from scratch.

        State must be the same size as the original state. State values can be
        either 0 or 1. There must be one and only one location that contains
        a value of 1. Whatever the numpy array type used, it will be converted
        to an integer numpy array.

        Parameters
        ----------
        initial_state: numpy.array
            The state to set the simulator to. If None then set to a random
            state.

        Raises
        ------
        ValueError
            If initial state's shape does not match (num_states, ). In
            otherwords the initial state must be a 1D numpy array with the
            same length as the existing state.
        ValueError
            If part of the state has a value or 1, or there are multiple
            parts of the state with value of 1.
        ValueError
            If there are values in the state other than 0 or 1.

        """
        if initial_state is None:
            self._state = QuadcopterDomain.__init_random_state(self.num_states)
        else:
            '''
            if initial_state.shape != (1, ):
                raise ValueError('The specified state did not match the '
                                 + 'current state size')
            state = initial_state.astype(np.int)
            if state[0] < 0 or state[0] >= self.num_states:
                raise ValueError('State value must be in range '
                                 + '[0, num_states)')
            '''
            self._state = initial_state

    def action_name(self, action):
        """Return string representation of actions.

        0:
            left
        1:
            right

        Returns
        -------
        str
            String representation of action.
        """
        return QuadcopterDomain.__action_names[action]

    

    def get_motor_speed_combination(self, action_index):
        index = action_index
        list_index = []
        motor_speeds = []
        while index != 0:
            mod = index % self.num_of_possible_speeds
            motor_speeds.append(self.possible_motor_speeds[mod])
            index = index // self.num_of_possible_speeds
        
        for i in range(4-len(motor_speeds)):
            motor_speeds.append(self.possible_motor_speeds[0])
        
        return motor_speeds[::-1]

    def set_target_state(self, position, lin_velocity=[0,0,0]):
        self.target_position = position
        self.target_lin_vel = lin_velocity

    def calc_reward(self, position, lin_vel, orientation):
        dist_reward = -10*position[2] - 4*(abs(position[0]) + abs(position[1])) - abs(orientation[0])-abs(orientation[1])-abs(orientation[2])
        vel_reward =  10*lin_vel[2] - 4*(abs(lin_vel[0]) + abs(lin_vel[1]))
        tot_reward = dist_reward + vel_reward
        if abs(position[2] - self.target_position[2]) <= 0.1:
            tot_reward += 50

        return np.tanh(tot_reward) 



        

    @staticmethod
    def __init_random_state(num_states):
        """Return randomly initialized state of the specified size."""
        #return np.array([randint(0, num_states-1)])
        return np.random.uniform(0.0, 1.0, size = (1, num_states))
        
