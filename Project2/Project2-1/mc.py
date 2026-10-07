#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 16:11:22 2019

@author: huiminren
# Modified By Yanhua Li on 08/19/2023 for gymnasium==0.29.0
# Updated for DS551/CS551 Fall 2026 with gymnasium==1.2.2
"""
import numpy as np
import random
from collections import defaultdict
#-------------------------------------------------------------------------
'''
    Monte-Carlo
    In this problem, you will implememnt an AI player for Blackjack.
    The main goal of this problem is to get familar with Monte-Carlo algorithm.

    You could test the correctness of your code
    by typing 'python -m nose -v mc_test.py' in the terminal.
'''
#-------------------------------------------------------------------------


def initial_policy(observation):
    """A policy that sticks if the player score is >= 20 and hit otherwise

    Parameters:
    -----------
    observation:
    Returns:
    --------
    action: 0 or 1
        0: STICK
        1: HIT
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    #                          #
    ############################
    if observation[0] >= 20:
        action = 0
    else: 
        action = 1
    return action


def mc_prediction(policy, env, n_episodes, gamma=1.0):
    """Given policy using sampling to calculate the value function
        by using Monte Carlo first visit algorithm.

    Parameters:
    -----------
    policy: function
        A function that maps an obversation to action probabilities
    env: function
        Gymnasium environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    Returns:
    --------
    V: defaultdict(float)
        A dictionary that maps from state to value
    """
    # initialize empty dictionaries
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> value
    V = defaultdict(float)

    ############################
    # YOUR IMPLEMENTATION HERE #
    #                          #
    ############################
    for i in range(n_episodes):
        episode = []
        state, info = env.reset()
        done = False

        while not done:
            action = policy(state)

            next_state, reward, terminated, truncated, info = env.step(action)

            episode.append((state, reward))

            done = terminated or truncated
            state = next_state

        G = 0

        for t in range(len(episode) - 1, -1, -1):
            state, reward = episode[t]

            G = gamma * G + reward

            earlier_states = [
                episode[j][0]
                for j in range(t)
            ]

            if state not in earlier_states:
                returns_sum[state] += G
                returns_count[state] += 1
                V[state] = returns_sum[state] / returns_count[state]

    return V


def epsilon_greedy(Q, state, nA, epsilon=0.1):
    """Selects epsilon-greedy action for supplied state.

    Parameters:
    -----------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    state: 
        current state
    nA: int
        Number of actions in the environment
    epsilon: float
        The probability to select a random action, range between 0 and 1

    Returns:
    --------
    action: int
        action based current state
    Hints:
    ------
    With probability (1 - epsilon) choose the greedy action.
    With probability epsilon choose an action at random.
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    #                          #
    ############################
    choice = np.random.choice(2, p=[(1-epsilon), epsilon])
    if np.random.rand() < epsilon:
        action = np.random.randint(nA)
    else:
        action = int(np.argmax(Q[state]))
    return action


def mc_control_epsilon_greedy(env, n_episodes, gamma=1.0, epsilon=0.1):
    """Monte Carlo control with exploring starts.
        Find an optimal epsilon-greedy policy.

    Parameters:
    -----------
    env: function
        Gymnasium environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    Hint:
    -----
    You could consider decaying epsilon, i.e. epsilon = epsilon-0.1/n_episode during each episode
    and episode must > 0.
    """

    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    ############################
    # YOUR IMPLEMENTATION HERE #
    #                          #
    ############################

    for i in range(n_episodes):
        state, info = env.reset()
        episode = []
        done = False


        while not done:
            action = epsilon_greedy(Q, state, env.action_space.n, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            episode.append((state, action, reward))

            state = next_state
            done = truncated or terminated

        G = 0

        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = G * gamma + reward

            earlier_state_actions = [
                (episode[j][0], episode[j][1])
                for j in range(t)
            ]

            if (state, action) not in earlier_state_actions:
                returns_sum[(state, action)] += G
                returns_count[(state, action)] += 1

                Q[state][action] = (returns_sum[(state, action)]/ returns_count[(state, action)])

        epsilon = max(0, epsilon - 0.1 / n_episodes)

    return Q
