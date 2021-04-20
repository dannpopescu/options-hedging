from ddpg import DDPG

ps = [
    {"name": "vgn", "plr": 0.0001, "vlr": 0.0001, "vgn": 0, "pgn": 'inf', "tau": 0.0001, "bs": 128},
    {"name": "vgn", "plr": 0.0001, "vlr": 0.0001, "vgn": 0.5, "pgn": 'inf', "tau": 0.0001, "bs": 128},
    {"name": "vgn", "plr": 0.0001, "vlr": 0.0001, "vgn": 1, "pgn": 'inf', "tau": 0.0001, "bs": 128},
    {"name": "vgn", "plr": 0.0001, "vlr": 0.0001, "vgn": 10, "pgn": 'inf', "tau": 0.0001, "bs": 128},
    {"name": "vgn", "plr": 0.0001, "vlr": 0.0001, "vgn": 100, "pgn": 'inf', "tau": 0.0001, "bs": 128},
    {"name": "vgn", "plr": 0.0001, "vlr": 0.0001, "vgn": 'inf', "pgn": 'inf', "tau": 0.0001, "bs": 128}
]

for v in ps:
    print("Start", v)
    ddpg = DDPG(17, v)
    ddpg.train(episodes=2000)


