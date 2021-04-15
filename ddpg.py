import gc
import random
import time
from itertools import count

import numpy as np
import torch

from const import LEAVE_PRINT_EVERY_N_SECS, ERASE_LINE, MAX_EPISODES, MAX_MINUTES
from env import HedgingEnv


class DDPG():
    def __init__(self,
                 replay_buffer_fn,
                 per_beta_schedule_fn,
                 policy_model_fn,
                 policy_max_grad_norm,
                 policy_optimizer_fn,
                 policy_optimizer_lr,
                 value_model_fn,
                 value_max_grad_norm,
                 value_optimizer_fn,
                 value_optimizer_lr,
                 training_strategy_fn,
                 evaluation_strategy_fn,
                 batch_size,
                 n_warmup_batches,
                 update_target_every_steps,
                 tau,
                 gamma):
        self.replay_buffer_fn = replay_buffer_fn
        self.per_beta_schedule_fn = per_beta_schedule_fn

        self.policy_model_fn = policy_model_fn
        self.policy_max_grad_norm = policy_max_grad_norm
        self.policy_optimizer_fn = policy_optimizer_fn
        self.policy_optimizer_lr = policy_optimizer_lr

        self.value_model_fn = value_model_fn
        self.value_max_grad_norm = value_max_grad_norm
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr

        self.training_strategy_fn = training_strategy_fn
        self.evaluation_strategy_fn = evaluation_strategy_fn

        self.batch_size = batch_size
        self.n_warmup_batches = n_warmup_batches
        self.min_samples = batch_size * n_warmup_batches

        self.update_target_every_steps = update_target_every_steps
        self.tau = tau

        self.gamma = gamma

        self.is_equals = []

    def optimize_model(self, experiences, weights=None, idxes=None):
        states, actions, rewards, next_states, is_terminals = experiences

        argmax_a_q_sp = self.target_policy_model(next_states)
        max_a_q_sp = self.target_value_model(next_states, argmax_a_q_sp)
        target_q_sa = rewards + self.gamma * max_a_q_sp * (1 - is_terminals)
        q_sa = self.online_value_model(states, actions)
        td_error = q_sa - target_q_sa.detach()
        if weights is not None:
            value_loss = (weights * td_error).pow(2).mul(0.5).mean()
        else:
            value_loss = td_error.pow(2).mul(0.5).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        # clip the gradients to obtain the Huber Loss.
        # self.value_max_grad_norm can be virtually any value, but know that this interacts with other hyperparameters,
        # such as learning rate
        torch.nn.utils.clip_grad_norm_(self.online_value_model.parameters(),
                                       self.value_max_grad_norm)
        self.value_optimizer.step()
        if idxes is not None:
            priorities = np.abs(td_error.detach().cpu().numpy() + 1e-10)  # 1e-10 to avoid zero priority
            self.replay_buffer.update_priorities(idxes, priorities)

        argmax_a_q_s = self.online_policy_model(states)
        max_a_q_s = self.online_value_model(states, argmax_a_q_s)
        policy_loss = -max_a_q_s.mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_policy_model.parameters(),
                                       self.policy_max_grad_norm)
        self.policy_optimizer.step()

    def interaction_step(self, state):
        action = self.training_strategy.select_action(self.online_policy_model,
                                                      state,
                                                      self.env)

        new_state, reward, is_terminal, info = self.env.step(action)
        self.replay_buffer.add(state, action, reward, new_state, is_terminal)
        self.episode_reward[-1] += reward
        self.episode_timestep[-1] += 1
        # self.episode_exploration[-1] += self.training_strategy.ratio_noise_injected
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

    def train(self, seed):
        training_start, last_debug_time = time.time(), float('-inf')

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

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

        nS, nA = 3, 1
        action_bounds = self.env.action_space.low, self.env.action_space.high

        self.episode_timestep = []
        self.episode_reward = []
        self.episode_seconds = []
        self.evaluation_scores = []
        self.episode_exploration = []

        self.target_value_model = self.value_model_fn(nS, nA)
        self.online_value_model = self.value_model_fn(nS, nA)
        self.target_policy_model = self.policy_model_fn(nS, action_bounds)
        self.online_policy_model = self.policy_model_fn(nS, action_bounds)
        self.update_networks(tau=1.0)
        self.value_optimizer = self.value_optimizer_fn(self.online_value_model,
                                                       self.value_optimizer_lr)
        self.policy_optimizer = self.policy_optimizer_fn(self.online_policy_model,
                                                         self.policy_optimizer_lr)

        self.replay_buffer = self.replay_buffer_fn()
        self.per_beta_schedule = self.per_beta_schedule_fn()
        self.training_strategy = self.training_strategy_fn()
        self.evaluation_strategy = self.evaluation_strategy_fn()

        result = np.empty((MAX_EPISODES, 5))
        result[:] = np.nan
        training_time = 0
        for episode in range(MAX_EPISODES):

            self.t = episode
            episode_start = time.time()

            state, is_terminal = self.env.reset(), False
            self.episode_reward.append(0.0)
            self.episode_timestep.append(0.0)
            self.episode_exploration.append(0.0)

            while True:
                state, is_terminal = self.interaction_step(state)

                if len(self.replay_buffer) > self.min_samples:
                    replay_buffer_sample = self.replay_buffer.sample(self.batch_size,
                                                                     beta=self.per_beta_schedule.value(self.t))
                    *experiences, weights, idxes = replay_buffer_sample
                    experiences = self.online_value_model.load(experiences)
                    self.optimize_model(experiences, weights, idxes)

                if np.sum(self.episode_timestep) % self.update_target_every_steps == 0:
                    self.update_networks()

                if is_terminal:
                    gc.collect()
                    break

            # stats
            episode_elapsed = time.time() - episode_start
            self.episode_seconds.append(episode_elapsed)
            training_time += episode_elapsed
            evaluation_score, _ = self.evaluate(self.online_policy_model, self.env)

            total_step = int(np.sum(self.episode_timestep))
            self.evaluation_scores.append(evaluation_score)

            mean_10_reward = np.mean(self.episode_reward[-10:])
            std_10_reward = np.std(self.episode_reward[-10:])
            mean_100_reward = np.mean(self.episode_reward[-100:])
            std_100_reward = np.std(self.episode_reward[-100:])
            # mean_100_eval_score = 0
            mean_100_eval_score = np.mean(self.evaluation_scores[-100:])
            # std_100_eval_score = 0
            std_100_eval_score = np.std(self.evaluation_scores[-100:])
            lst_100_exp_rat = np.array(
                self.episode_exploration[-100:] ) /np.array(self.episode_timestep[-100:])
            mean_100_exp_rat = np.mean(lst_100_exp_rat)
            std_100_exp_rat = np.std(lst_100_exp_rat)

            wallclock_elapsed = time.time() - training_start
            result[episode-1] = total_step, mean_100_reward, \
                                mean_100_eval_score, training_time, wallclock_elapsed

            reached_debug_time = time.time() - last_debug_time >= LEAVE_PRINT_EVERY_N_SECS
            reached_max_minutes = wallclock_elapsed >= MAX_MINUTES * 60
            reached_max_episodes = episode >= MAX_EPISODES
            reached_goal_mean_reward = mean_100_eval_score >= 0
            training_is_over = reached_max_minutes or \
                               reached_max_episodes or \
                               reached_goal_mean_reward
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - training_start))
            debug_message = 'el {}, ep {:04}, ts {:07}, '
            debug_message += 'ar 10 {:05.1f}\u00B1{:05.1f}, '
            debug_message += '100 {:05.1f}\u00B1{:05.1f}, '
            debug_message += 'ex 100 {:02.1f}\u00B1{:02.1f}, '
            debug_message += 'ev {:05.1f}\u00B1{:05.1f}'
            debug_message = debug_message.format(
                elapsed_str, episode-1, total_step, mean_10_reward, std_10_reward,
                mean_100_reward, std_100_reward, mean_100_exp_rat, std_100_exp_rat,
                mean_100_eval_score, std_100_eval_score)
            print(debug_message, end='\r', flush=True)
            if reached_debug_time or training_is_over:
                print(ERASE_LINE + debug_message, flush=True)
                last_debug_time = time.time()
            if training_is_over:
                if reached_max_minutes: print(u'--> reached_max_minutes \u2715')
                if reached_max_episodes: print(u'--> reached_max_episodes \u2715')
                if reached_goal_mean_reward: print(u'--> reached_goal_mean_reward \u2713')
                break

        final_eval_score, score_std = self.evaluate(self.online_policy_model, self.env, n_episodes=100)
        wallclock_time = time.time() - training_start
        print('Training complete.')
        print('Final evaluation score {:.2f}\u00B1{:.2f} in {:.2f}s training time,'
              ' {:.2f}s wall-clock time.\n'.format(
            final_eval_score, score_std, training_time, wallclock_time))
        self.env.close() ; del self.env
        return result, final_eval_score, training_time, wallclock_time

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
