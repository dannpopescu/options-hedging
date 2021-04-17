from ddpg import DDPG

ps = [
    {"name": "plr", "plr": 0.0001, "vlr": 0.0001, "pgn": 'inf', "vgn": 'inf', "tau": 0.0001, "bs": 128},
    {"name": "plr", "plr": 0.0005, "vlr": 0.0001, "pgn": 'inf', "vgn": 'inf', "tau": 0.0001, "bs": 128},
    {"name": "plr", "plr": 0.0010, "vlr": 0.0001, "pgn": 'inf', "vgn": 'inf', "tau": 0.0001, "bs": 128},
    {"name": "plr", "plr": 0.0015, "vlr": 0.0001, "pgn": 'inf', "vgn": 'inf', "tau": 0.0001, "bs": 128},
    {"name": "plr", "plr": 0.0020, "vlr": 0.0001, "pgn": 'inf', "vgn": 'inf', "tau": 0.0001, "bs": 128},
    {"name": "plr", "plr": 0.0025, "vlr": 0.0001, "pgn": 'inf', "vgn": 'inf', "tau": 0.0001, "bs": 128},
    {"name": "plr", "plr": 0.0030, "vlr": 0.0001, "pgn": 'inf', "vgn": 'inf', "tau": 0.0001, "bs": 128},
    {"name": "plr", "plr": 0.0050, "vlr": 0.0001, "pgn": 'inf', "vgn": 'inf', "tau": 0.0001, "bs": 128},
]

for v in ps:
    print("Start", v)
    ddpg = DDPG(17, v)
    ddpg.train(episodes=10)


