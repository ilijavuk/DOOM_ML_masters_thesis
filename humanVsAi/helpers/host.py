import os
import vizdoom as vzd

def startHost(configName):
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, configName))

    game.set_doom_map("map01")

    # Host game with options that will be used in the competition.
    game.add_game_args("-host 2 "  
                    # This machine will function as a host for a multiplayer game with this many players (including this machine). 
                    # It will wait for other machines to connect using the -join parameter and then start the game when everyone is connected.
                    "-port 5029 "              # Specifies the port (default is 5029).
                    "+viz_connect_timeout 60 " # Specifies the time (in seconds), that the host will wait for other players (default is 60).
                    "-deathmatch "             # Deathmatch rules are used for the game.
                    "+timelimit 10.0 "         # The game (episode) will end after this many minutes have elapsed.
                    "+sv_forcerespawn 1 "      # Players will respawn automatically after they die.
                    "+sv_noautoaim 1 "         # Autoaim is disabled for all players.
                    "+sv_respawnprotect 1 "    # Players will be invulnerable for two second after spawning.
                    "+sv_s4pawnfarthest 1 "     # Players will be spawned as far as possible from any other players.
                    "+sv_nocrouch 1 "          # Disables crouching.
                    "+viz_respawn_delay 2 "   # Sets delay between respawns (in seconds, default is 0).
                    "+viz_nocheat 1")          # Disables depth and labels buffer and the ability to use commands that could interfere with multiplayer game.

    # This can be used to host game without taking part in it (can be simply added as argument of vizdoom executable).
    #game.add_game_args("+viz_spectator 1")

    # colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
    game.add_game_args("+name Player +colorset 3")

    game.set_mode(vzd.Mode.ASYNC_SPECTATOR)

    game.set_screen_resolution(vzd.ScreenResolution.RES_1920X1080)

    game.init()

    while not game.is_episode_finished():
        s = game.get_state()

        # Taking an initial action - the game crashes otherwise
        game.make_action([1,0,0,0,0,0,0,0,0])
        
        if game.is_player_dead():
            game.respawn_player()

    game.close()