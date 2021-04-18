from ddpg import DDPG

ps = [
    {"name": "vlrvgn", "vlrvgn": "0.002-10", "plr": 0.0001, "vlr": 0.002, "pgn": 'inf', "vgn": 10, "tau": 0.0001, "bs": 128},
    {"name": "vlrvgn", "vlrvgn": "0.002-0.5", "plr": 0.0001, "vlr": 0.002, "pgn": 'inf', "vgn": 0.5, "tau": 0.0001, "bs": 128},
]

for v in ps:
    print("Start", v)
    ddpg = DDPG(17, v)
    ddpg.train(episodes=2000)


