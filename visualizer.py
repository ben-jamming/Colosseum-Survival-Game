import world
import ui
import matplotlib.pyplot as plt
from agents import utils

if __name__ == "__main__":
    world_1 = world.World(board_size=5, display_delay=1)
    player = world_1.p0_pos
    adversary = world_1.p1_pos
    max_step = world_1.max_step
    is_player_turn = world_1.turn == 0

    ui_engine = ui.UIEngine(5, world=world_1)
    state = {
        "board": world_1.chess_board,
        "player": tuple(player),
        "adversary": tuple(adversary),
        "max_step": max_step,
        "is_player_turn": True
    }


    moves_p0 = utils.get_possible_positions(state)
    state["is_player_turn"] = False
    moves_p1 = utils.get_possible_positions(state)

    is_terminal_state = utils.is_terminal(state)
    scores = utils.score(state)
    p0_score = scores[0]
    p1_score = scores[1]
    player_territory, adversary_territory = utils.simple_territory_search(state)
    
    print("p0_score: ", p0_score)
    print("p1_score: ", p1_score)

    print("player_territory: ", len(player_territory))
    print("adversary_territory: ", len(adversary_territory))
    # print(is_terminal_state)
    # print(explored)

    while True:
      # if the window closes, stop the simulation
      if not plt.fignum_exists(1):
        break
      ui_engine.render(world_1.chess_board,
                        tuple(player),
                        tuple(adversary),
                        valid_moves_p0=player_territory,
                        valid_moves_p1=adversary_territory
      )
      # check for a keyboard interrupt of if the window is closed
      try:
        plt.pause(0.1)
      except KeyboardInterrupt:
        break
      except:
        break