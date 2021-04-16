from ddpg import DDPG

ddpg = DDPG(seed=17)

result, final_eval_score, training_time, wallclock_time = ddpg.train(episodes=50_000)
