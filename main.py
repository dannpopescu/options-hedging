from ddpg import DDPG

ddpg = DDPG(seed=1)

ddpg.train(episodes=50_000)