from .utils import perform_action, undo_last_action, score
import time
import numpy as np

class AlphaBeta:
    


    def get_action(generate_children, 
                   utility, 
                   state, 
                   max_depth, 
                   time_limit=0.5,
                   breadth_limit=10,
                   ):
        """
        children returned from generate children is a list of actiosn
        actions are (position, wall_direction)
        """
        start_time = time.time()

        def child_heuristic(child):
          player_pos = state['player']
          child_pos = child[0]
          distance_from_player = np.linalg.norm(np.array(player_pos) - np.array(child_pos))
          distance_to_adv = np.linalg.norm(np.array(child_pos) - np.array(state['adversary']))
          return distance_from_player * distance_to_adv  + distance_from_player
        
        def minimax(cur_state, depth, alpha, beta):
            # if max_player = True, then we are maximizing player
            current_time = time.time()
            if current_time - start_time > time_limit:
                return utility(cur_state)
            children = generate_children(cur_state)
            children.sort(key=child_heuristic)
            children = children[:int(breadth_limit/depth)]


            if depth == max_depth or not children:
                return utility(cur_state)
            
            max_player = cur_state['is_player_turn']

            cur_val = float('-inf') if max_player else float('inf')

            for child in children:
                if current_time - start_time > time_limit:
                    break
                
                perform_action(cur_state, child)
                val = minimax(cur_state, depth + 1, alpha, beta)
                undo_last_action(cur_state)

                if max_player:
                    cur_val = max(cur_val, val)
                    alpha = max(alpha, cur_val)
                else: 
                    cur_val = min(cur_val, val)
                    beta = min(beta, cur_val)
                if beta <= alpha:
                    break
            return cur_val


        # get the move that maximizes the utility
        children = generate_children(state)
        # sort each child by its distance from the player
            
        # the sort function sorts in 
        children.sort(key=child_heuristic)
        children = children[:breadth_limit]

            

        max_child = children[0]
        max_val = float('-inf')
        for child in children:
            perform_action(state, child)
            child_val = minimax(state, 1, float('-inf'), float('inf'))
            undo_last_action(state)
            if child_val > max_val:
                max_val = child_val
                max_child = child
        
        def get_utility(move):
            perform_action(state, move)
            val = utility(state)
            undo_last_action(state)
            return val
        
        #print(f'chosen action: {max_child}, val: {max_val}, utility: {get_utility(max_child)}')
        if max_val == -1:
            for child in children:
                perform_action(state, child)
                child_val = utility(state)
                undo_last_action(state)
                if child_val > max_val:
                    max_val = child_val
                    max_child = child

        perform_action(state, max_child)
        # player_score, adv_score = score(state)
        # print(f'chosen action: {max_child}, val: {max_val} scores: p:{player_score}, a:{adv_score})')
        undo_last_action(state)

        return max_child