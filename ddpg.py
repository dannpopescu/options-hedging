import gc
import copy
import os
import random
import time
from itertools import count

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from actor import Actor
from buffer import ReplayBuffer
from critic import Critic
from env import HedgingEnv
from const import LEAVE_PRINT_EVERY_N_SECS, ERASE_LINE
from strategy import EGreedyExpStrategy, GreedyStrategy, NormalNoiseStrategy
from baselines.schedules import LinearSchedule
from baselines.replay_buffer import PrioritizedReplayBuffer


class DDPG():
    def __init__(self, seed):

        self.writer = SummaryWriter("logdir")
        # self.writer = SummaryWriter("logs/" + ps["name"] + str(ps[ps["name"]]))
        self.evaluation_step = 0

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Trading environment
        self.env = HedgingEnv(init_price=100,
                              mu=0.05,
                              sigma=0.2,
                              strike_price=100,
                              r=0,
                              q=0,
                              trading_freq=1,
                              maturity=1/12,
                              trading_cost=0.01)
        self.env.seed(seed)
        self.env.action_space.seed(seed)

        action_bounds = self.env.action_space.low, self.env.action_space.high
        state_space, action_space = 3, 1

        # Policy model - actor
        self.actor = Actor(state_dim=state_space, action_dim=action_space, action_bounds=action_bounds)
        self.actor_target = copy.deepcopy(self.actor)

        # Value model - critic
        self.critic = Critic(state_dim=state_space, action_dim=action_space)
        self.critic_target = copy.deepcopy(self.critic)

        # Use Huber loss: 0 - MAE, inf - MSE
        self.actor_max_grad_norm = float("inf")
        self.critic_max_grad_norm = float("inf")

        # Use Polyak averaging - mix the target network with a fraction of online network
        self.tau = 0.0001
        self.update_target_every_steps = 1

        # Optimizers
        self.actor_optimizer = Adam(params=self.actor.parameters(), lr=1e-4, eps=1e-7)
        self.critic_q1_optimizer = Adam(params=self.critic.q1.parameters(), lr=0.0025, eps=1e-7)
        self.critic_q2_optimizer = Adam(params=self.critic.q2.parameters(),  lr=0.0025, eps=1e-7)

        # Use Prioritized Experience Replay - PER as the replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(size=600_000,
                                                     alpha=0.6)
        self.per_beta_schedule = LinearSchedule(schedule_timesteps=50_000,
                                                final_p=1.0,
                                                initial_p=0.4)

        # Training strategy
        self.training_strategy = EGreedyExpStrategy(epsilon=1,
                                                    min_epsilon=0.1,
                                                    epsilon_decay=0.9999)
        self.evaluation_strategy = GreedyStrategy()

        self.batch_size = 128
        self.gamma = 1

        # total iterations
        self.total_optimizations = 0
        self.total_steps = 0
        self.total_ev_interactions = 0

        self.q1_loss = []
        self.q2_loss = []
        self.actor_loss = []

        self.mean_a_grad = 0
        self.std_a_grad = 0

        self.mean_weights = 0
        self.std_weights = 0

    def optimize_model(self, experiences, weights, idxs):
        self.total_optimizations += 1
        self.optimize_critic(experiences, weights, idxs)
        self.optimize_actor(experiences)

    def optimize_critic(self, experiences, weights, idxs):
        states, actions, rewards, next_states, is_terminals = experiences
        weights = torch.tensor(weights, dtype=torch.float32, device=self.critic.device).unsqueeze(1)

        next_actions = self.actor_target(next_states)

        next_values_1 = self.critic_target.Q1(next_states, next_actions)
        next_values_2 = self.critic_target.Q2(next_states, next_actions)

        done_mask = 1 - is_terminals

        target_1 = rewards + self.gamma * next_values_1 * done_mask
        target_2 = rewards ** 2 \
                    + (self.gamma ** 2 * next_values_2) * done_mask \
                    + (2 * self.gamma * rewards * next_values_1) * done_mask

        td_error_1 = self.critic.Q1(states, actions) - target_1.detach()
        critic_q1_loss = (weights * td_error_1 ** 2).mean()
        # optimize critic 1
        self.critic_q1_optimizer.zero_grad()
        critic_q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.q1.parameters(),
                                       self.critic_max_grad_norm)
        self.critic_q1_optimizer.step()


        td_error_2 = self.critic.Q2(states, actions) - target_2.detach()
        critic_q2_loss = (weights * td_error_2 ** 2).mean()
        # optimize critic Q2
        self.critic_q2_optimizer.zero_grad()
        critic_q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.q2.parameters(),
                                       self.critic_max_grad_norm)
        self.critic_q2_optimizer.step()

        # update priorities in replay buffer
        priorities = (np.abs(td_error_2.detach().cpu().numpy()) + 1e-10).flatten()  # 1e-10 to avoid zero priority
        self.replay_buffer.update_priorities(idxs, priorities)

        self.q1_loss.append(td_error_1.detach().pow(2).cpu().numpy().mean())
        self.q2_loss.append(td_error_2.detach().pow(2).cpu().numpy().mean())

        # self.writer.add_scalar("critic_q1_loss", critic_q1_loss.detach().cpu().numpy(), self.total_optimizations)
        # self.writer.add_scalar("critic_q2_loss", critic_q2_loss.detach().cpu().numpy(), self.total_optimizations)

    def optimize_actor(self, experiences):
        states, actions, rewards, next_states, is_terminals = experiences

        chosen_actions = self.actor(states)
        chosen_actions.retain_grad()

        expected_reward = self.critic(states, chosen_actions)
        actor_loss = -expected_reward.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(),
                                       self.actor_max_grad_norm)
        self.actor_optimizer.step()

        self.mean_a_grad = np.mean(chosen_actions.grad)
        self.std_a_grad = np.std(chosen_actions.grad)
        self.actor_loss.append(float(actor_loss.detach().cpu()))

        # self.writer.add_scalar("actor_loss", actor_loss.detach().cpu().numpy(), self.total_optimizations)

    def interaction_step(self, state):
        self.total_steps += 1

        action, is_exploratory = self.training_strategy.select_action(self.actor,
                                                                      state,
                                                                      self.env)
        new_state, reward, is_terminal, info = self.env.step(action)
        self.replay_buffer.add(state, action, reward, new_state, is_terminal)

        self.episode_reward[-1] += reward
        self.episode_exploration[-1] += int(is_exploratory)

        return new_state, is_terminal

    def update_networks(self):
        self.mix_weights(target_model=self.critic_target.q1, online_model=self.critic.q1)
        self.mix_weights(target_model=self.critic_target.q2, online_model=self.critic.q2)
        self.mix_weights(target_model=self.actor_target, online_model=self.actor)

    def mix_weights(self, target_model, online_model):
        for target_param, online_param in zip(target_model.parameters(), online_model.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1 - self.tau) * target_param.data)

    def train(self, episodes):
        training_start, last_debug_time = time.time(), float('-inf')

        self.episode_reward = []
        self.episode_exploration = []
        self.episode_seconds = []

        result = np.empty((episodes, 4))
        result[:] = np.nan
        training_time = 0

        for episode in range(1, episodes+1):
            episode_start = time.time()

            state, is_terminal = self.env.reset(), False

            self.path_length = self.env.simulator.days_to_maturity()
            self.episode_reward.append(0.0)
            self.episode_exploration.append(0.0)

            for step in count():
                state, is_terminal = self.interaction_step(state)

                if len(self.replay_buffer) > self.batch_size:
                    *experiences, weights, idxs = self.replay_buffer.sample(self.batch_size,
                                                                            beta=self.per_beta_schedule.value(episode))

                    self.mean_weights = np.mean(weights)
                    self.std_weights = np.std(weights)

                    experiences = self.critic.load(experiences)
                    self.optimize_model(experiences, weights, idxs)

                    if step % self.update_target_every_steps == 0:
                        self.update_networks()

                if is_terminal:
                    gc.collect()
                    break

            self.training_strategy.epsilon_update()

            # Stats

            # elapsed time
            episode_elapsed = time.time() - episode_start
            self.episode_seconds.append(episode_elapsed)
            training_time += episode_elapsed
            wallclock_elapsed = time.time() - training_start

            reached_debug_time = time.time() - last_debug_time >= LEAVE_PRINT_EVERY_N_SECS

            if len(self.q1_loss) >= 100:
                elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - training_start))
                msg = 'el {}, ep {:>5}, Q1 lst {:>5.0f}, 100 {:>5.0f}\u00B1{:04.0f}, ' \
                      + 'Q2 lst {:>10.0f}, 100 {:>10.0f}\u00B1{:09.0f}, ' \
                      + 'A lst {:05.1f}, 100 {:05.1f}\u00B1{:05.1f}'
                msg = msg.format(
                    elapsed_str, episode,
                    self.q1_loss[-1], np.mean(self.q1_loss[-100:]), np.std(self.q1_loss[-100:]),
                    self.q2_loss[-1], np.mean(self.q2_loss[-100:]), np.std(self.q2_loss[-100:]),
                    self.actor_loss[-1], np.mean(self.actor_loss[-100:]), np.std(self.actor_loss[-100:]))
                print(msg, end='\r', flush=True)
                if reached_debug_time or episode >= episodes:
                    print(ERASE_LINE + msg, flush=True)
                    last_debug_time = time.time()

                if episode % 50 == 0:
                    hist = {
                        "episode": [episode],
                        "last_q1_loss": [self.q1_loss[-1]],
                        "mean_q1_loss": [np.mean(self.q1_loss)],
                        "std_q1_loss": [np.std(self.q1_loss)],
                        "last_q2_loss": [self.q2_loss[-1]],
                        "mean_q2_loss": [np.mean(self.q2_loss)],
                        "std_q2_loss": [np.std(self.q2_loss)],
                        "last_actor_loss": [self.actor_loss[-1]],
                        "mean_actor_loss": [np.mean(self.actor_loss)],
                        "std_actor_loss": [np.std(self.actor_loss)],
                        "mean_weights": [self.mean_weights],
                        "std_weights": [self.std_weights],
                        "mean_a_grad": [self.mean_a_grad],
                        "std_a_grad": [self.std_a_grad],
                    }
                    hist_path = "history/metrics_hist.csv"
                    if not os.path.exists(hist_path):
                        pd.DataFrame.from_dict(hist).to_csv(hist_path, index=False, encoding='utf-8')
                    else:
                        pd.DataFrame.from_dict(hist).to_csv(hist_path, mode='a', index=False, header=False, encoding='utf-8')

                    if episode % 300 == 0:
                        self.q1_loss = self.q1_loss[-100:]
                        self.q2_loss = self.q2_loss[-100:]
                        self.actor_loss = self.actor_loss[-100:]

            # tensorboard metrics
            # self.writer.add_scalar("epsilon", self.training_strategy.epsilon, episode)

            # if episode % 10 == 0 and episode != 0:
                # self.evaluate(self.actor, self.env)

            if episode % 100 == 0:
                filename = 'model/ddpg_' + str(int(episode / 100)) + ".pt"
                self.save(episode, filename)

    def save(self, episode, filename):
        torch.save({
            'episode': episode,
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_q1': self.critic.q1.state_dict(),
            'critic_target_q1': self.critic_target.q1.state_dict(),
            'critic_q1_optimizer': self.critic_q1_optimizer.state_dict(),
            'critic_q2': self.critic.q2.state_dict(),
            'critic_target_q2': self.critic_target.q2.state_dict(),
            'critic_q2_optimizer': self.critic_q2_optimizer.state_dict(),
        }, filename)

    def load(self, filename):
        saved = torch.load(filename)
        self.actor.load_state_dict(saved['actor'])
        self.actor_target.load_state_dict(saved['actor_target'])
        self.actor_optimizer.load_state_dict(saved['actor_optimizer'])
        self.critic.q1.load_state_dict(saved['critic_q1'])
        self.critic_target.q2.load_state_dict(saved['critic_target_q1'])
        self.critic.q2.load_state_dict(saved['critic_q2'])
        self.critic_target.q2.load_state_dict(saved['critic_target_q2'])
        self.critic_q2_optimizer.load_state_dict(saved['critic_q2_optimizer'])

    def test(self, episodes):
        model_actions = []
        model_rewards = []
        model_final_rewards = []

        delta_actions = []
        delta_rewards = []
        delta_final_rewards = []

        for i in range(1, episodes + 1):
            state, done = self.env.reset(), False
            while not done:
                action = self.evaluation_strategy.select_action(self.actor, state)
                state, reward, done, info = self.env.step(action)
                model_actions.append(action)
                model_rewards.append(reward)
                delta_actions.append(info["delta_action"])
                delta_rewards.append(info["delta_reward"])
            model_final_rewards.append(np.sum(model_rewards))
            delta_final_rewards.append(np.sum(delta_rewards))
            model_rewards = []
            delta_rewards = []

            if i % 1000 == 0:
                print("{:0>5}: model {:.2f}  {:.2f}   delta {:.2f}  {:.2f}".format(i,
                                                               np.mean(model_final_rewards), np.std(model_final_rewards),
                                                               np.mean(delta_final_rewards), np.std(delta_final_rewards)))

    def evaluate(self, eval_policy_model, eval_env, n_episodes=1):
        actions = []
        rewards = []
        delta_actions = []
        delta_rewards = []
        for _ in range(n_episodes):
            self.evaluation_step += 1
            s, d = eval_env.reset(), False
            for _ in count():
                self.total_ev_interactions += 1
                a = self.evaluation_strategy.select_action(eval_policy_model, s)
                s, r, d, i = eval_env.step(a)
                actions.append(a)
                rewards.append(r)
                delta_actions.append(i["delta_action"])
                delta_rewards.append(i["delta_reward"])
                self.writer.add_scalars("ev_actions", {"actor": a}, self.total_ev_interactions)
                self.writer.add_scalars("ev_actions", {"delta": i["delta_action"]}, self.total_ev_interactions)
                if d: break
        diffs = np.array(actions) - np.array(delta_actions)
        diffs_mean = np.mean(diffs)
        diffs_std = np.std(diffs)

        self.writer.add_scalars("ev", {"actor_reward": np.sum(rewards)}, self.evaluation_step)
        self.writer.add_scalars("ev", {"delta_reward": np.sum(delta_rewards)}, self.evaluation_step)

        self.writer.add_scalars("ev_diff", {"mean": diffs_mean}, self.evaluation_step)
        self.writer.add_scalars("ev_diff", {"std": diffs_std}, self.evaluation_step)

        self.writer.flush()
