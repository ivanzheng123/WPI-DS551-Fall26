#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim

from agent import Agent
from dqn_model import DQN
"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)


class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(Agent_DQN,self).__init__(env)
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.qNet = DQN(num_actions = env.action_space.n).to(self.device)
        self.targetNet = DQN(num_actions = env.action_space.n).to(self.device)

        self.targetNet.load_state_dict(
            self.qNet.state_dict()
        )

        self.optimizer = optim.Adam(self.qNet.parameters(), lr = .0001)

        self.memory = deque(maxlen = 10000)

        self.gamma = 0.99
        self.batch_size = 32
        self.epsilon = 1.0
        self.num_episodes = 40000

        self.steps = 0
        
        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #
            weights = torch.load("dqn_model.pth", map_location=self.device, weights_only=True)
            self.qNet.load_state_dict(weights)

            self.qNet.eval()
            

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        ###########################
        pass
    
    
    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        if not test and random.random() < self.epsilon:
            return self.env.action_space.sample()

        state = torch.as_tensor(observation, device=self.device)
        state = state.permute(2,0,1).unsqueeze(0)

        with torch.no_grad():
            q_vals = self.qNet(state)
            action = q_vals.argmax(dim=1).item()
        ###########################
        return action
    
    def push(self, state, action, reward, next_state, done):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        
        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        transition = (np.array(state, copy=True), int(action), float(reward), np.array(next_state, copy=True), bool(done))
        self.memory.append(transition)
        ###########################
        
        
    def replay_buffer(self):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.as_tensor(np.stack(states), device=self.device).permute(0, 3, 1, 2)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        next_states = torch.as_tensor(np.stack(next_states), device=self.device).permute(0, 3, 1, 2)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        ###########################
        return states, actions, rewards, next_states, dones
        

    def train(self):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        total_rewards = 0
        for episode in range(self.num_episodes):
            self.init_game_setting()
            result = self.env.reset()

            state = result[0] if isinstance(result, tuple) else result
            done = False
            total_reward = 0

            while not done:
                action = self.make_action(state, test=False)
                result = self.env.step(action)

                if len(result) == 5:
                    next_state, reward, terminated, truncated, _ = result
                    done = terminated or truncated
                    terminal = terminated
                else:
                    next_state, reward, done, _ = result
                    terminal = done

                self.push(state, action, reward, next_state, terminal)

                state = next_state
                total_reward += reward
                self.steps += 1

                self.epsilon = max(0.1, 1.0 - 0.9 * self.steps/1000000)

                if len(self.memory) >= self.batch_size and self.steps % 4 == 0:
                    states, actions, rewards, next_states, dones = (self.replay_buffer())

                    q = self.qNet(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        next_q = self.targetNet(next_states).max(1).values
                        target = (rewards + self.gamma * (1 - dones) * next_q)

                    loss = F.smooth_l1_loss(q, target)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    if self.steps % 1000 == 0:
                        self.targetNet.load_state_dict(self.qNet.state_dict())
            total_rewards += total_reward
            if (episode + 1) % 100 == 0:
                torch.save(self.qNet.state_dict(), "dqn_model.pth")

            print(f"Episode {episode + 1}: Reward = {total_reward}")
        print(f"Average Reward: {total_rewards / self.num_episodes} Steps: {self.steps}")
        torch.save(self.qNet.state_dict(), "dqn_model.pth")
        ###########################
