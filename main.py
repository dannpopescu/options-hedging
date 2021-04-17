from ddpg import DDPG

# ddpg = DDPG(seed=17, ps)

result, final_eval_score, training_time, wallclock_time = ddpg.train(episodes=50_000)

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

ps = [
    {"name": "bs", "plr": 0.0001, "vlr": 0.0001, "pgn": 'inf', "vgn": 'inf', "tau": 0.0001, "bs": 64},
    {"name": "bs", "plr": 0.0001, "vlr": 0.0001, "pgn": 'inf', "vgn": 'inf', "tau": 0.0001, "bs": 128},
    {"name": "bs", "plr": 0.0001, "vlr": 0.0001, "pgn": 'inf', "vgn": 'inf', "tau": 0.0001, "bs": 256},
    {"name": "bs", "plr": 0.0001, "vlr": 0.0001, "pgn": 'inf', "vgn": 'inf', "tau": 0.0001, "bs": 512}
]

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