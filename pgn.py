from ddpg import DDPG

ps = [
    {"name": "pgn", "plr": 0.0001, "vlr": 0.0001, "pgn": 0, "vgn": 'inf', "tau": 0.0001, "bs": 128},
    {"name": "pgn", "plr": 0.0001, "vlr": 0.0001, "pgn": 0.5, "vgn": 'inf', "tau": 0.0001, "bs": 128},
    {"name": "pgn", "plr": 0.0001, "vlr": 0.0001, "pgn": 1, "vgn": 'inf', "tau": 0.0001, "bs": 128},
    {"name": "pgn", "plr": 0.0001, "vlr": 0.0001, "pgn": 10, "vgn": 'inf', "tau": 0.0001, "bs": 128},
    {"name": "pgn", "plr": 0.0001, "vlr": 0.0001, "pgn": 100, "vgn": 'inf', "tau": 0.0001, "bs": 128},
    {"name": "pgn", "plr": 0.0001, "vlr": 0.0001, "pgn": 'inf', "vgn": 'inf', "tau": 0.0001, "bs": 128}
]

for v in ps:
    print("Start", v)
    ddpg = DDPG(17, v)
    ddpg.train(episodes=2000)


