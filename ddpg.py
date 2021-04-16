import gc
import random
import time
from itertools import count

import numpy as np
import torch
from torch.optim import Adam

from actor import Actor
from baselines.replay_buffer import PrioritizedReplayBuffer
from baselines.schedules import LinearSchedule
from const import LEAVE_PRINT_EVERY_N_SECS, ERASE_LINE, MAX_EPISODES, MAX_MINUTES
from critic import Critic
from env import HedgingEnv

from strategy import EGreedyExpStrategy, GreedyStrategy


class DDPG():
    def __init__(self, seed):

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
                              maturity=0.5,
                              trading_cost=0.01,
                              seed=seed)

        action_bounds = self.env.action_space.low, self.env.action_space.high
        state_space, action_space = 3, 1

        # Policy model - actor
        self.target_policy_model = Actor(input_dim=state_space,
                                         output_dim=action_space,
                                         action_bounds=action_bounds)
        self.online_policy_model = Actor(input_dim=state_space,
                                         output_dim=action_space,
                                         action_bounds=action_bounds)

        # Value model - critic
        self.target_value_model = Critic(input_dim=state_space + action_space)
        self.online_value_model = Critic(input_dim=state_space + action_space)

        # Use Huber loss: 0 - MAE, inf - MSE
        self.policy_max_grad_norm = float('inf')
        self.value_max_grad_norm = float('inf')

        # Copy networks' parameters from online to target
        self.update_networks(tau=1.0)

        # Use Polyak averaging - mix the target network with a fraction of online network
        self.tau = 0.0001
        self.update_target_every_steps = 1

        # Optimizers
        self.policy_optimizer = Adam(params=self.online_policy_model.parameters(),
                                     lr=1e-4)
        self.value_optimizer = Adam(params=self.online_value_model.parameters(),
                                    lr=1e-4)

        # Use Prioritized Experience Replay - PER as the replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(size=600_000,
                                                     alpha=0.6)
        self.per_beta_schedule = LinearSchedule(schedule_timesteps=50_000,
                                                final_p=1.0,
                                                initial_p=0.4)

        # Training strategy
        self.training_strategy = EGreedyExpStrategy(init_epsilon=1,
                                                    min_epsilon=0.1,
                                                    epsilon_decay=0.99994)
        self.evaluation_strategy = GreedyStrategy()

        self.batch_size = 128
        self.gamma = 0

    def optimize_model(self, experiences, weights, idxs):
        states, actions, rewards, next_states, is_terminals = experiences

        argmax_a_q_sp = self.target_policy_model(next_states)
        max_a_q_sp = self.target_value_model(next_states, argmax_a_q_sp)
        target_q_sa = rewards + self.gamma * max_a_q_sp * (1 - is_terminals)
        q_sa = self.online_value_model(states, actions)
        td_error = q_sa - target_q_sa.detach()
        weights = torch.tensor(weights, dtype=torch.float32, device=self.target_value_model.device).unsqueeze(1)
        value_loss = (weights * td_error).pow(2).mul(0.5).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_value_model.parameters(),
                                       self.value_max_grad_norm)
        self.value_optimizer.step()
        priorities = np.abs(td_error.detach().cpu().numpy() + 1e-10)  # 1e-10 to avoid zero priority
        self.replay_buffer.update_priorities(idxs, priorities)

        argmax_a_q_s = self.online_policy_model(states)
        max_a_q_s = self.online_value_model(states, argmax_a_q_s)
        policy_loss = -max_a_q_s.mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_policy_model.parameters(),
                                       self.policy_max_grad_norm)
        self.policy_optimizer.step()

    def interaction_step(self, state):
        action, is_exploratory = self.training_strategy.select_action(self.online_policy_model,
                                                                      state,
                                                                      self.env)
        new_state, reward, is_terminal, info = self.env.step(action)
        self.replay_buffer.add(state, action, reward, new_state, is_terminal)

        self.episode_reward[-1] += reward
        self.episode_exploration[-1] += int(is_exploratory)

        return new_state, is_terminal

    def update_networks(self, tau=None):
        if tau is None:
            tau = self.tau
        self.mix_weights(tau, self.target_value_model, self.online_value_model)
        self.mix_weights(tau, self.target_policy_model, self.online_policy_model)

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

        self.path_length = self.env.simulator.days_to_maturity()

        for episode in range(episodes):

            episode_start = time.time()

            state, is_terminal = self.env.reset(), False

            self.episode_reward.append(0.0)
            self.episode_exploration.append(0.0)

            for step in count():
                state, is_terminal = self.interaction_step(state)

                *experiences, weights, idxs = self.replay_buffer.sample(self.batch_size,
                                                                         beta=self.per_beta_schedule.value(episode))
                experiences = self.online_value_model.load(experiences)
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
            lst_100_exp_rat = np.array(np.array(self.episode_exploration[-100:]) / self.path_length)
            mean_100_exp_rat = np.mean(lst_100_exp_rat)
            std_100_exp_rat = np.std(lst_100_exp_rat)

            result[episode-1] = mean_100_reward, mean_100_exp_rat, training_time, wallclock_elapsed

            reached_debug_time = time.time() - last_debug_time >= LEAVE_PRINT_EVERY_N_SECS

            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - training_start))
            debug_message = 'el {}, ep {:04}, '
            debug_message += 'ar 10 {:05.1f}\u00B1{:05.1f}, '
            debug_message += '100 {:05.1f}\u00B1{:05.1f}, '
            debug_message += 'ex 100 {:02.1f}\u00B1{:02.1f}, '
            debug_message += 'ev {:05.1f}\u00B1{:05.1f}'
            debug_message = debug_message.format(
                elapsed_str, episode-1,
                mean_10_reward, std_10_reward,
                mean_100_reward, std_100_reward,
                mean_100_exp_rat, std_100_exp_rat)
            print(debug_message, end='\r', flush=True)
            if reached_debug_time or episode >= episodes:
                print(ERASE_LINE + debug_message, flush=True)
                last_debug_time = time.time()

            if episode % 1000 == 0:
                torch.save({
                    'episode': episode,
                    'online_policy_state_dict': self.online_policy_model.state_dict(),
                    'target_policy_state_dict': self.target_policy_model.state_dict(),
                    'online_value_state_dict': self.online_value_model.state_dict(),
                    'target_value_state_dict': self.target_value_model.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
                    'value_optimizer_state_dict': self.value_optimizer.state_dict(),
                }, 'model/ddpg' + str(int(episode / 1000)) + ".pt")

        return result, training_time, wallclock_elapsed

    def evaluate(self, eval_policy_model, eval_env, n_episodes=1):
        rs = []
        for _ in range(n_episodes):
            s, d = eval_env.reset(), False
            rs.append(0)
            for _ in count():
                a = self.evaluation_strategy.select_action(eval_policy_model, s)
                s, r, d, _ = eval_env.step(a)
                rs[-1] += r
                if d: break
        return np.mean(rs), np.std(rs)
