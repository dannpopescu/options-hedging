from ddpg import DDPG

ps = [
    {"name": "tau", "plr": 0.0001, "vlr": 0.0001, "pgn": 'inf', "vgn": 'inf', "tau": 0.0001, "bs": 128},
    {"name": "tau", "plr": 0.0001, "vlr": 0.0001, "pgn": 'inf', "vgn": 'inf', "tau": 0.00008, "bs": 128},
    {"name": "tau", "plr": 0.0001, "vlr": 0.0001, "pgn": 'inf', "vgn": 'inf', "tau": 0.00005, "bs": 128},
    {"name": "tau", "plr": 0.0001, "vlr": 0.0001, "pgn": 'inf', "vgn": 'inf', "tau": 0.00015, "bs": 128},
    {"name": "tau", "plr": 0.0001, "vlr": 0.0001, "pgn": 'inf', "vgn": 'inf', "tau": 0.00020, "bs": 128},
    {"name": "tau", "plr": 0.0001, "vlr": 0.0001, "pgn": 'inf', "vgn": 'inf', "tau": 0.00050, "bs": 128},
    {"name": "tau", "plr": 0.0001, "vlr": 0.0001, "pgn": 'inf', "vgn": 'inf', "tau": 0.00080, "bs": 128},
    {"name": "tau", "plr": 0.0001, "vlr": 0.0001, "pgn": 'inf', "vgn": 'inf', "tau": 0.0010, "bs": 128}
]

for v in ps:
    print("Start", v)
    ddpg = DDPG(17, v)
    ddpg.train(episodes=2000)


