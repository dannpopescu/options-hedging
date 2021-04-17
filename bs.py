from ddpg import DDPG

ps = [
    {"name": "bs", "plr": 0.0001, "vlr": 0.0001, "pgn": 'inf', "vgn": 'inf', "tau": 0.0001, "bs": 64},
    {"name": "bs", "plr": 0.0001, "vlr": 0.0001, "pgn": 'inf', "vgn": 'inf', "tau": 0.0001, "bs": 128},
    {"name": "bs", "plr": 0.0001, "vlr": 0.0001, "pgn": 'inf', "vgn": 'inf', "tau": 0.0001, "bs": 256},
    {"name": "bs", "plr": 0.0001, "vlr": 0.0001, "pgn": 'inf', "vgn": 'inf', "tau": 0.0001, "bs": 512}
]

for v in ps:
    print("Start", v)
    ddpg = DDPG(17, v)
    ddpg.train(episodes=2000)


