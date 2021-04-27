import gc
import copy
import random
import time
from itertools import count

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from actor import Actor
from critic import Critic
from env import HedgingEnv
from const import LEAVE_PRINT_EVERY_N_SECS, ERASE_LINE
from strategy import EGreedyExpStrategy, GreedyStrategy
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
        self.critic_1 = Critic(state_dim=state_space, action_dim=action_space)
        self.critic_2 = Critic(state_dim=state_space, action_dim=action_space)
        self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_2_target = copy.deepcopy(self.critic_2)

        # Use Huber loss: 0 - MAE, inf - MSE
        self.actor_max_grad_norm = float("inf")
        self.critic_max_grad_norm = float("inf")

        # Use Polyak averaging - mix the target network with a fraction of online network
        self.tau = 0.0001
        self.update_target_every_steps = 1

        # Optimizers
        self.actor_optimizer = Adam(params=self.actor.parameters(), lr=5e-4)
        self.critic_optimizer_1 = Adam(params=self.critic_1.parameters(), lr=1e-3)
        self.critic_optimizer_2 = Adam(params=self.critic_2.parameters(),  lr=1e-3)

        # Use Prioritized Experience Replay - PER as the replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(size=600_000,
                                                     alpha=0.6)
        self.per_beta_schedule = LinearSchedule(schedule_timesteps=50_000,
                                                final_p=1.0,
                                                initial_p=0.4)

        # Training strategy
        self.training_strategy = EGreedyExpStrategy(init_epsilon=1,
                                                    min_epsilon=0.1,
                                                    epsilon_decay=0.9999)
        self.evaluation_strategy = GreedyStrategy()

        self.batch_size = 128
        self.gamma = 1

        # total iterations
        self.total_optimizations = 0
        self.total_steps = 0
        self.total_ev_interactions = 0

    def optimize_model(self, experiences, weights, idxs):
        self.total_optimizations += 1
        self.optimize_critic(experiences, weights, idxs)
        self.optimize_actor(experiences)

    def optimize_critic(self, experiences, weights, idxs):
        states, actions, rewards, next_states, is_terminals = experiences

        next_actions = self.actor_target(next_states)

        next_values_1 = self.critic_1_target(next_states, next_actions)
        next_values_2 = self.critic_2_target(next_states, next_actions)

        done_mask = 1 - is_terminals

        target_1 = rewards + self.gamma * next_values_1 * done_mask
        target_2 = rewards ** 2 \
                    + (self.gamma ** 2 * next_values_2) * done_mask \
                    + (2 * self.gamma * rewards * next_values_1) * done_mask

        td_error_1 = target_1.detach() - self.critic_1(states, actions)
        td_error_2 = target_2.detach() - self.critic_2(states, actions)

        weights = torch.tensor(weights, dtype=torch.float32, device=self.critic_1_target.device).unsqueeze(1)

        critic_1_loss = (weights * td_error_1).pow(2).mul(0.5).mean()
        critic_2_loss = (weights * td_error_2).pow(2).mul(0.5).mean()

        # optimize critic 1
        self.critic_optimizer_1.zero_grad()
        critic_1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(),
                                       self.critic_max_grad_norm)
        self.critic_optimizer_1.step()

        # optimize critic Q2
        self.critic_optimizer_2.zero_grad()
        critic_2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(),
                                       self.critic_max_grad_norm)
        self.critic_optimizer_2.step()

        # update priorities in replay buffer
        priorities = np.abs(td_error_1.detach().cpu().numpy()) + 1e-10  # 1e-10 to avoid zero priority
        self.replay_buffer.update_priorities(idxs, priorities)

        self.writer.add_scalar("critic_1_loss", critic_1_loss.detach().cpu().numpy(), self.total_optimizations)
        self.writer.add_scalar("critic_2_loss", critic_2_loss.detach().cpu().numpy(), self.total_optimizations)

    def optimize_actor(self, experiences):
        states, actions, rewards, next_states, is_terminals = experiences

        chosen_actions = self.actor(states)

        # Objective Function: expected hedging cost plus a constant
        # multiplied by the std of the hedging cost
        q1 = self.critic_1(states, chosen_actions)
        q2 = self.critic_2(states, chosen_actions)
        values = q1 + 1.5 * torch.sqrt(q2 - q1*q1)

        actor_loss = -values.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(),
                                       self.actor_max_grad_norm)
        self.actor_optimizer.step()

        self.writer.add_scalar("actor_loss", actor_loss.detach().cpu().numpy(), self.total_optimizations)

    def interaction_step(self, state):
        self.total_steps += 1

        action, is_exploratory = self.training_strategy.select_action(self.actor,
                                                                      state,
                                                                      self.env)
        new_state, reward, is_terminal, info = self.env.step(action)
        self.replay_buffer.add(state, action, reward, new_state, is_terminal)

        self.episode_reward[-1] += reward
        self.episode_exploration[-1] += int(is_exploratory)

        self.writer.add_scalar("actions", action, self.total_steps)

        return new_state, is_terminal

    def update_networks(self, tau=None):
        if tau is None:
            tau = self.tau
        self.mix_weights(tau, self.critic_1_target, self.critic_1)
        self.mix_weights(tau, self.critic_2_target, self.critic_2)
        self.mix_weights(tau, self.actor_target, self.actor)

    def mix_weights(self, tau, target_model, online_model):
        for target, online in zip(target_model.parameters(),
                                  online_model.parameters()):
            target_ratio = (1.0 - tau) * target.data
            online_ratio = tau * online.data
            mixed_weights = target_ratio + online_ratio
            target.data.copy_(mixed_weights)

    def train(self, episodes):
        training_start, last_debug_time = time.time(), float('-inf')

        self.episode_reward = []
        self.episode_exploration = []
        self.episode_seconds = []

        result = np.empty((episodes, 4))
        result[:] = np.nan
        training_time = 0

        for episode in range(episodes):
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
                    experiences = self.critic_1.load(experiences)
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

            # reward
            mean_10_reward = np.mean(self.episode_reward[-10:])
            std_10_reward = np.std(self.episode_reward[-10:])
            mean_100_reward = np.mean(self.episode_reward[-100:])
            std_100_reward = np.std(self.episode_reward[-100:])

            # exploration rate
            lst_100_exp_rat = np.array(self.episode_exploration[-100:]) / self.path_length
            mean_100_exp_rat = np.mean(lst_100_exp_rat)
            std_100_exp_rat = np.std(lst_100_exp_rat)

            # tensorboard metrics
            self.writer.add_scalar("reward, 100-SMA", mean_100_reward, episode)
            self.writer.add_scalar("reward, 10-SMA", mean_10_reward, episode)
            self.writer.add_scalar("exploration rate", self.episode_exploration[-1] / self.path_length, episode)
            self.writer.add_scalar("epsilon", self.training_strategy.epsilon, episode)

            if episode % 10 == 0 and episode != 0:
                self.evaluate(self.actor, self.env)

            result[episode-1] = mean_100_reward, mean_100_exp_rat, training_time, wallclock_elapsed

            reached_debug_time = time.time() - last_debug_time >= LEAVE_PRINT_EVERY_N_SECS

            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - training_start))
            debug_message = 'el {}, ep {:04}, '
            debug_message += 'ar 10 {:05.1f}\u00B1{:05.1f}, '
            debug_message += '100 {:05.1f}\u00B1{:05.1f}, '
            debug_message += 'ex 100 {:02.1f}\u00B1{:02.1f}, '
            debug_message = debug_message.format(
                elapsed_str, episode-1,
                mean_10_reward, std_10_reward,
                mean_100_reward, std_100_reward,
                mean_100_exp_rat, std_100_exp_rat)
            print(debug_message, end='\r', flush=True)
            if reached_debug_time or episode >= episodes:
                print(ERASE_LINE + debug_message, flush=True)
                last_debug_time = time.time()

            if episode % 1000 == 0 and episode != 0:
                torch.save({
                    'episode': episode,
                    'online_policy_state_dict': self.online_actor.state_dict(),
                    'target_policy_state_dict': self.target_actor.state_dict(),
                    'online_value_state_dict': self.online_critic.state_dict(),
                    'target_value_state_dict': self.target_critic.state_dict(),
                    'policy_optimizer_state_dict': self.actor_optimizer.state_dict(),
                    'value_optimizer_state_dict': self.critic_optimizer.state_dict(),
                }, 'model/ddpg_' + str(int(episode / 1000)) + ".pt")

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