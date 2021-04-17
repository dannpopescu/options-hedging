from ddpg import DDPG

ps = [
    {"name": "vlr", "plr": 0.0001, "vlr": 0.0001, "pgn": 'inf', "vgn": 'inf', "tau": 0.0001, "bs": 128},
    {"name": "vlr", "plr": 0.0001, "vlr": 0.0005, "pgn": 'inf', "vgn": 'inf', "tau": 0.0001, "bs": 128},
    {"name": "vlr", "plr": 0.0001, "vlr": 0.0008, "pgn": 'inf', "vgn": 'inf', "tau": 0.0001, "bs": 128},
    {"name": "vlr", "plr": 0.0001, "vlr": 0.0010, "pgn": 'inf', "vgn": 'inf', "tau": 0.0001, "bs": 128},
    {"name": "vlr", "plr": 0.0001, "vlr": 0.0020, "pgn": 'inf', "vgn": 'inf', "tau": 0.0001, "bs": 128},
    {"name": "vlr", "plr": 0.0001, "vlr": 0.0030, "pgn": 'inf', "vgn": 'inf', "tau": 0.0001, "bs": 128},
    {"name": "vlr", "plr": 0.0001, "vlr": 0.0050, "pgn": 'inf', "vgn": 'inf', "tau": 0.0001, "bs": 128},
    {"name": "vlr", "plr": 0.0001, "vlr": 0.0070, "pgn": 'inf', "vgn": 'inf', "tau": 0.0001, "bs": 128},
]

for v in ps:
    print("Start", v)
    ddpg = DDPG(17, v)
    ddpg.train(episodes=2000)


