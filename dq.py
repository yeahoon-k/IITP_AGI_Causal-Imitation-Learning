import sys
from collections import deque
from typing import Optional

import gym
import numpy as np
import tensorflow as tf
from causallearn.search.ConstraintBased.PC import pc
from joblib import delayed, Parallel
from tensorflow import keras


# define neural net Q_\theta(s,a) as a class
class Qfunction(keras.Model):
    def __init__(self, obssize, actsize, hidden_dims):
        """
        obssize: dimension of state space
        actsize: dimension of action space
        hidden_dims: list containing output dimension of hidden layers
        """
        super(Qfunction, self).__init__()

        # Layer weight initializer
        initializer = keras.initializers.RandomUniform(minval=-1., maxval=1.)

        # Input Layer
        self.input_layer = keras.layers.InputLayer(input_shape=(obssize,))

        # Hidden Layer
        self.hidden_layers = []
        for hidden_dim in hidden_dims:
            # TODO: define each hidden layers. e.g., hidden_dims = [10,5]
            # hidden_layer = [[hidden 1 layer],[hidden 2 layer]]
            hidden = keras.layers.Dense(hidden_dim, activation="relu")
            self.hidden_layers.append(hidden)  # None

        # Output Layer :
        # TODO: Define the output layer.
        # state -> DNN model -> action_pred
        self.output_layer = keras.layers.Dense(actsize)  # None

    @tf.function
    def call(self, states):
        # TODO: You SHOULD implement the model's forward pass
        # forward step
        x = self.input_layer(states)
        x = self.hidden_layers[0](x)
        x = self.hidden_layers[1](x)

        return self.output_layer(x)
        # return None


# Wrapper class for training Qfunction and updating weights (target network)
class DQN(object):
    def __init__(self, obssize, actsize, hidden_dims, optimizer):
        """
        obssize: dimension of state space
        actsize: dimension of action space
        optimizer:
        """
        # 관측 사이즈, 액션사즈, 히든 디멘젼
        self.qfunction = Qfunction(obssize, actsize, hidden_dims)
        self.optimizer = optimizer
        self.obssize = obssize
        self.actsize = actsize

    def _predict_q(self, states, actions):
        """
        states represent s_t
        actions represent a_t
        """
        # TODO: Define the logic for calculate  Q_\theta(s,a)
        onehot = tf.one_hot(indices=actions, depth=self.actsize, \
                            dtype=tf.float32)

        # Qvalue_forward = self.qfunction(states)
        Qvalue_forward = self.compute_Qvalues(states)
        return tf.math.reduce_sum(tf.math.multiply(onehot, Qvalue_forward), 1)

    def _loss(self, Qpreds, targets):
        """
        Qpreds represent Q_\theta(s,a)
        targets represent the terms E[r+gamma Q] in Bellman equations

        This function is OBJECTIVE function
        """
        return tf.math.reduce_mean(tf.square(Qpreds - targets))

    def compute_Qvalues(self, states):
        """
        states: numpy array as input to the neural net, states should have
        size [numsamples, obssize], where numsamples is the number of samples
        output: Q values for these states. The output should have size
        [numsamples, actsize] as numpy array
        """
        inputs = np.atleast_2d(states.astype('float32'))
        return self.qfunction(inputs)

    def train(self, states, actions, targets):
        """
        states: numpy array as input to compute loss (s)
        actions: numpy array as input to compute loss (a)
        targets: numpy array as input to compute loss (Q targets)
        """
        with tf.GradientTape() as tape:
            # forward 해서 현재 state, behavior action의 output
            Qpreds = self._predict_q(states, actions)
            # target Q값을 목표로 loss 계산하여 업데이트
            loss = self._loss(Qpreds, targets)
        variables = self.qfunction.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    def update_weights(self, from_network):
        """
        We need a subroutine to update target network
        i.e. to copy from principal network to target network.
        This function is for copying  𝜃←𝜃target
        """

        from_var = from_network.qfunction.trainable_variables
        to_var = self.qfunction.trainable_variables

        for v1, v2 in zip(from_var, to_var):
            v2.assign(v1)


# Wrapper class for training Qfunction and updating weights (target network)
class DQN_imitator(object):
    def __init__(self, obssize, actsize, hidden_dims, optimizer):
        """
        obssize: dimension of state space
        actsize: dimension of action space
        optimizer:
        """
        # 관측 사이즈, 액션사즈, 히든 디멘젼
        self.qfunction = Qfunction(obssize, actsize, hidden_dims)
        self.optimizer = optimizer
        self.obssize = obssize
        self.actsize = actsize

    def _predict_q(self, states, actions):
        """
        states represent s_t
        actions represent a_t
        """
        # TODO: Define the logic for calculate  Q_\theta(s,a)
        onehot = tf.one_hot(indices=actions, depth=self.actsize, \
                            dtype=tf.float32)

        # Qvalue_forward = self.qfunction(states)
        Qvalue_forward = self.compute_Qvalues(states)
        return tf.math.reduce_sum(tf.math.multiply(onehot, Qvalue_forward), 1)

    def _loss(self, Qpreds, targets):
        """
        Qpreds represent Q_\theta(s,a)
        targets represent the terms E[r+gamma Q] in Bellman equations

        This function is OBJECTIVE function
        """
        return tf.math.reduce_mean(tf.square(Qpreds - targets))

    def compute_Qvalues(self, states):
        """
        states: numpy array as input to the neural net, states should have
        size [numsamples, obssize], where numsamples is the number of samples
        output: Q values for these states. The output should have size
        [numsamples, actsize] as numpy array
        """
        inputs = np.atleast_2d(states.astype('float32'))
        return self.qfunction(inputs)

    def train(self, states, actions, targets):
        """
        states: numpy array as input to compute loss (s)
        actions: numpy array as input to compute loss (a)
        targets: numpy array as input to compute loss (Q targets)
        """
        with tf.GradientTape() as tape:
            # forward 해서 현재 state, behavior action의 output
            Qpreds = self._predict_q(states, actions)
            # target Q값을 목표로 loss 계산하여 업데이트
            loss = self._loss(Qpreds, targets)
        variables = self.qfunction.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    def update_weights(self, from_network):
        """
        We need a subroutine to update target network
        i.e. to copy from principal network to target network.
        This function is for copying  𝜃←𝜃target
        """
        from_var = from_network.qfunction.trainable_variables
        to_var = self.qfunction.trainable_variables

        for v1, v2 in zip(from_var, to_var):
            v2.assign(v1)


# Implement replay buffer
# import random for minibatch
import random


class ReplayBuffer(object):
    def __init__(self, maxlength):
        """
        maxlength: max number of tuples to store in the buffer
        if there are more tuples than maxlength, pop out the oldest tuples
        """
        self.buffer = deque()
        self.number = 0
        self.maxlength = maxlength

    def replace(self, new_buffer: deque) -> 'ReplayBuffer':
        self.buffer = new_buffer
        while len(self.buffer) > self.maxlength:
            self.buffer.popleft()
        self.number = len(self.buffer)
        return self

    def append(self, experience):
        """
        this function implements appending new experience tuple
        experience: a tuple of the form (s,a,r,s^\prime)
        """
        self.buffer.append(experience)
        self.number += 1

    def pop(self):
        """
        pop out the oldest tuples if self.number > self.maxlength
        """
        while self.number > self.maxlength:
            self.buffer.popleft()
            self.number -= 1

    def sample(self, batchsize):
        """
        this function samples 'batchsize' experience tuples
        batchsize: size of the minibatch to be sampled
        return: a list of tuples of form (s,a,r,s^\prime)
        """

        # sample from buffer(population)
        sampled_data = random.sample(self.buffer, batchsize)
        return sampled_data  # need implementation

    def __iter__(self):
        return iter(self.buffer)

    def copy(self):
        return ReplayBuffer(self.maxlength).replace(deque(self.buffer))


default_params = {
    'lr': 1e-3,  # learning rate for gradient update
    'batchsize': 64,  # batchsize for buffer sampling
    'maxlength': 10000,  # max number of tuples held by buffer
    'envname': "CartPole-v0",  # environment name
    'tau': 100,  # time steps for target update
    'episodes': 1600,  # number of episodes to run
    'initialsize': 500,  # initial time steps before start updating
    'epsilon': .05,  # constant for exploration
    'gamma': .99,  # discount
    'hidden_dims': [10, 5]  # hidden dimensions
}


def train(lr=1e-3  # learning rate for gradient update
          , batchsize=64  # batchsize for buffer sampling
          , maxlength=10000  # max number of tuples held by buffer
          , envname="CartPole-v0"  # environment name
          , tau=100  # time steps for target update
          , episodes=1600  # number of episodes to run
          , initialsize=500  # initial time steps before start updating
          , epsilon=.05  # constant for exploration
          , gamma=.99  # discount
          , hidden_dims=[10, 5]  # hidden dimensions
          , perform_range=500
          , replay_buffer: Optional[ReplayBuffer] = None
          , causal=False,
          seed=None,
          noisy=0
          ):
    # initialize environment

    if seed is not None:
        random.seed(seed)
        tf.random.set_seed(seed)
        np.random.seed(seed)

    env = gym.make(envname)
    obssize = env.observation_space.low.size + noisy
    state_selector = list(range(obssize))
    actsize = env.action_space.n

    # optimizer
    optimizer = keras.optimizers.Adam(learning_rate=lr)

    # initialization of buffer
    if replay_buffer is None:
        replay_buffer = ReplayBuffer(maxlength)
    elif noisy:
        inner_buffer = replay_buffer.buffer
        new_inner_buffer = deque()
        for sample in inner_buffer:
            new_sample0 = np.concatenate((sample[0], np.random.randn(noisy)))
            new_sample4 = np.concatenate((sample[4], np.random.randn(noisy)))
            new_inner_buffer.append((new_sample0, *sample[1:4], new_sample4))

        replay_buffer = ReplayBuffer(maxlength).replace(new_inner_buffer)

    if causal:
        assert replay_buffer is not None

        data = long_data = np.array([_[0] for _ in replay_buffer])
        if len(data) > 500:
            data = data[:500, :]
        cg = pc(data, indep_test='kci', alpha=0.01, kernelX='Gaussian', kernelY='Gaussian', kernelZ='Gaussian', approx=True, est_width='median')
        print(cg.G.graph)
        state_selector = list(np.nonzero(np.sum(np.abs(cg.G.graph), axis=0))[0])
        print(state_selector)
        selected_replay_buffer = deque()
        for state, row in zip(long_data[:, state_selector], replay_buffer):
            new_row = (state, *row[1:-1], row[-1][state_selector])
            selected_replay_buffer.append(new_row)

        replay_buffer = ReplayBuffer(maxlength).replace(selected_replay_buffer)
        obssize = len(state_selector)

    # initialize networks
    Qprincipal = DQN(obssize, actsize, hidden_dims, optimizer)
    Qtarget = DQN(obssize, actsize, hidden_dims, optimizer)
    Qtarget.update_weights(Qprincipal)

    # main iteration
    rrecord = []
    totalstep = 0
    for ite in range(episodes):

        try:
            obs, *_ = env.reset(seed=ite)
        except TypeError:
            obs, *_ = env.reset()

        if noisy:
            obs = np.concatenate((obs, np.random.randn(noisy)))
        if causal:
            obs = obs[state_selector]
        done = False
        rsum = 0

        while not done:
            # Exploration based on 'epsilon = .05 (constant for exploration)'
            # epsilon greedy
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                # 1 - epsilon
                Q_val = Qprincipal.compute_Qvalues(np.expand_dims(obs, 0))
                # print(Q_val), (E.g., Q_val : [1,2,10,4,5])
                action = np.argmax(Q_val)

            # sample from environment using selected action
            new_obs, r, done, *_ = env.step(action)
            if noisy:
                new_obs = np.concatenate((new_obs, np.random.randn(noisy)))

            if causal:
                new_obs = new_obs[state_selector]

            # make tuple new_ex using old&new environment values
            done_ = 1 if done else 0
            new_ex = (obs, action, r, done_, new_obs)

            # append new_ex to buffuer
            replay_buffer.append(new_ex)

            # pop oldest tuple if buffer is full
            while replay_buffer.number > maxlength:
                replay_buffer.pop()

            # increase step & cumulate reward
            totalstep += 1
            rsum += r

            # replace obs value with new_obs
            obs = new_obs

            # sample minibatch and train
            # update weight when totalstep is greater than initialsize
            if totalstep > initialsize:
                sample_s, sample_a, sample_r, sample_done, sample_ns = [], [], [], [], []
                minibatch = replay_buffer.sample(batchsize)
                for sample in minibatch:
                    sample_s.append(sample[0])
                    sample_a.append(sample[1])
                    sample_r.append(sample[2])
                    sample_done.append(sample[3])
                    sample_ns.append(sample[4])

                # list to numpy
                sample_s, sample_a, sample_r, sample_done, sample_ns = \
                    np.array(sample_s), np.array(sample_a), np.array(sample_r), \
                    np.array(sample_done), np.array(sample_ns)

                target_predQ = np.max(Qtarget.compute_Qvalues(sample_ns), 1)
                target_Q = sample_r + gamma * target_predQ * (1 - sample_done)

                # train
                Qprincipal.train(sample_s, sample_a, target_Q)

                if totalstep % tau == 0:
                    Qtarget.update_weights(Qprincipal)
            pass

        rrecord.append(rsum)
        if ite % 10 == 0:
            print('iteration {} ave reward {}'.format(ite, np.mean(rrecord[-10:])))

    print('-----------')
    return replay_buffer, sum(rrecord[-perform_range:]) if perform_range < episodes else sum(rrecord)


def main(n_epi, seed):
    print(f'with episodes {n_epi}')
    expert_buffer, expert_perf = train(episodes=n_epi, seed=seed)
    _, noncausal_imitator_perf = train(episodes=n_epi, replay_buffer=expert_buffer.copy(), seed=seed, noisy=3)
    _, causal_imitator_perf = train(episodes=n_epi, replay_buffer=expert_buffer.copy(), causal=True, seed=seed, noisy=3, )
    return expert_perf, noncausal_imitator_perf, causal_imitator_perf

    # X = np.random.randn(1000, 5)
    # X[:,0] = 0.5 * X[:,1] + 0.3 * X[:,0]
    # cg = pc(X, indep_test='kci', kernelX='Gaussian', kernelY='Gaussian', kernelZ='Gaussian', approx=True, est_width='median')
    # # cg = pc(X, indep_test='fisherz')
    # print(cg.G.graph)


if __name__ == '__main__':
    n_epi = int(sys.argv[1]) if len(sys.argv) >= 2 else 200
    outs = Parallel(5)(delayed(main)(n_epi, _) for _ in range(20))
    xx = result = np.array([list(_) for _ in outs])
    print(result)
    print(np.mean(xx, axis=0))
    print(np.sum(xx[:, 2] >= xx[:, 1], axis=0))
    print(np.mean(xx[:, 2] / xx[:, 1], axis=0))
    print(np.sum(xx[:, 2]) / np.sum(xx[:, 1]))
    #
    # xx = np.array([[38931., 4920., 54710.],
    #                [2089., 2365., 2710.],
    #                [2740., 4086., 2501.],
    #                [1962., 2918., 2005.],
    #                [26265., 2605., 22312.],
    #                [30370., 4800., 6921.],
    #                [23146., 2987., 8793.],
    #                [31514., 2892., 3876.],
    #                [2163., 2820., 3337.],
    #                [4848., 2646., 3321.],
    #                [8084., 3998., 11918.],
    #                [4745., 26132., 26289.],
    #                [2458., 3669., 3070.],
    #                [2009., 2672., 4349.],
    #                [36692., 2364., 2749.],
    #                [2006., 3812., 2868.],
    #                [3211., 2678., 2392.],
    #                [11175., 14888., 18952.],
    #                [13254., 3055., 7423.],
    #                [3754., 16719., 26106.],
    #                [14458., 4208., 48159.],
    #                [2888., 3396., 2939.],
    #                [19223., 3101., 15495.],
    #                [2030., 2323., 2082.],
    #                [49087., 3305., 4269.],
    #                [4182., 5231., 4388.],
    #                [2129., 2166., 2042.],
    #                [2042., 2568., 2209.],
    #                [1989., 2986., 3806.],
    #                [2067., 2087., 2127.]])
    # print(np.mean(xx, axis=0))
    # print(np.sum(xx[:, 2] >= xx[:, 1], axis=0))
    # print(np.mean(xx[:, 2] / xx[:, 1], axis=0))
    # print(np.sum(xx[:, 2]) / np.sum(xx[:, 1]))
