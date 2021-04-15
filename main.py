from torch import optim
import numpy as np

from actor import Actor
from critic import Critic
from strategy import GreedyStrategy
from strategy import EGreedyExpStrategy
from ddpg import DDPG
from baselines.replay_buffer import PrioritizedReplayBuffer
from baselines.schedules import LinearSchedule

SEEDS = (17,)


ddpg_results = []
best_agent, best_eval_score = None, float('-inf')
# for seed in SEEDS:
seed = SEEDS[0]
environment_settings = {
    'gamma': 1,
    'max_minutes': 600,
    'max_episodes': 1000000,
    'goal_mean_100_reward': 10000000
}

policy_model_fn = lambda nS, bounds: Actor(nS, 1, bounds)
policy_max_grad_norm = float('inf')
policy_optimizer_fn = lambda net, lr: optim.Adam(net.parameters(), lr=lr)
policy_optimizer_lr = 1e-3

value_model_fn = lambda nS, nA: Critic(nS + nA)
value_max_grad_norm = float('inf')
value_optimizer_fn = lambda net, lr: optim.Adam(net.parameters(), lr=lr)
value_optimizer_lr = 0.001

# training_strategy_fn = lambda bounds: NormalNoiseStrategy(bounds, exploration_noise_ratio=0.1)
training_strategy_fn = lambda: EGreedyExpStrategy(init_epsilon=1, min_epsilon=0.1, epsilon_decay=0.99994)
evaluation_strategy_fn = lambda: GreedyStrategy()

# replay_buffer_fn = lambda: ReplayBuffer(max_size=100000, batch_size=256)
replay_buffer_alpha = 0.6
replay_buffer_fn = lambda: PrioritizedReplayBuffer(500_000, replay_buffer_alpha)
replay_buffer_init_beta = 0.4
replay_buffer_beta_steps = 50000
replay_buffer_beta_schedule_fn = lambda: LinearSchedule(replay_buffer_beta_steps,
                                                     final_p=1.0,
                                                     initial_p=replay_buffer_init_beta)

batch_size = 128

n_warmup_batches = 5
update_target_every_steps = 1
tau = 0.001

gamma, max_minutes, max_episodes, goal_mean_100_reward = environment_settings.values()

agent = DDPG(replay_buffer_fn,
             replay_buffer_beta_schedule_fn,
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
             gamma)

result, final_eval_score, training_time, wallclock_time = agent.train(seed)

ddpg_results.append(result)
# if final_eval_score > best_eval_score:
#     best_eval_score = final_eval_score
#     best_agent = agent
ddpg_results = np.array(ddpg_results)
# _ = BEEP()