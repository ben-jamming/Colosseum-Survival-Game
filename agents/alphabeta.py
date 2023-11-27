from .utils import perform_action, undo_last_action, score


class AlphaBeta:
    


    def get_action(generate_children, utility, state, max_depth):
        """
        children returned from generate children is a list of actiosn
        actions are (position, wall_direction)
        """
        
        def minimax(cur_state, depth, alpha, beta):
            # if max_player = True, then we are maximizing player
            if depth == max_depth or not generate_children(cur_state):
                return utility(cur_state)
            
            max_player = cur_state['is_player_turn']

            cur_val = float('-inf') if max_player else float('inf')
            for child in generate_children(cur_state):
                
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
        max_child = children[0]
        max_val = float('-inf')
        for child in children:
            perform_action(state, child)
            child_val = minimax(state, 1, float('-inf'), float('inf'))
            undo_last_action(state)
            if child_val > max_val:
                max_val = child_val
                max_child = child
        
        if max_val == float('-inf'):
            for child in children:
                perform_action(state, child)
                child_val = utility(state)
                undo_last_action(state)
                if child_val > max_val:
                    max_val = child_val
                    max_child = child

        perform_action(state, max_child)
        player_score, adv_score = score(state)
        # print(f'chosen action: {max_child}, val: {max_val} scores: p:{player_score}, a:{adv_score})')
        undo_last_action(state)

        return max_child