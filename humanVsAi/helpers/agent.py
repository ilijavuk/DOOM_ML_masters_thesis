import os
from stable_baselines3 import PPO
import vizdoom as vzd
from humanVsAi.helpers.process_frame import process_frame
import numpy as np

def startAgent(configName, modelPath):
    game = vzd.DoomGame()

    # Use CIG example config or your own.
    game.load_config(os.path.join(vzd.scenarios_path, configName))

    game.set_doom_map("map01")
    
    # Join existing game.
    game.add_game_args("-join 127.0.0.1 -port 5029") # Connect to a host for a multiplayer game.

    # Name your agent and select color
    # colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
    game.add_game_args("+name AI +colorset 0")

    game.set_mode(vzd.Mode.ASYNC_PLAYER)

    game.set_window_visible(False)

    game.init()

    model = PPO.load(modelPath)
    action_number = 4
    actions = np.identity(action_number, dtype=np.uint8)

    while not game.is_episode_finished():
        s = game.get_state()
        obs = process_frame(s.screen_buffer)

        action, _ = model.predict(obs)
        print(f"Action {action}")
        game.make_action(actions[action])
        
        if game.is_player_dead():
            game.respawn_player()

    game.close()