class AlphaBeta:
    def get_action(generate_children, utility, state, max_depth):
        def minimax(cur_state, depth, alpha, beta, max_player):
            # if max_player = True, then we are maximizing player
            if depth == max_depth or not generate_children(cur_state):
                return utility(cur_state) 
            
            cur_val = float('-inf') if max_player else float('inf')
            for child in generate_children(cur_state):
                val = minimax(child, depth + 1, alpha, beta, not max_player)
                if max_player:
                    cur_val = max(cur_val, val)
                    alpha = max(alpha, cur_val)
                else: 
                    cur_val = min(cur_val, val)
                    beta = min(beta, cur_val)
                if beta <= alpha:
                    break
            return cur_val
      
        children = generate_children(state)
        # remove states where the player doesn't move
        # for child in children:
        #     print(f'child: {child["player"]}')
            
        # print(f'children: {children}')
        vals = [minimax(child, 0, float('-inf'), float('inf'), False) for child in children]
        child_vals = list(zip(children, vals))
        # for child, val in child_vals:
        #     print(f'move: {child["player"]}, val: {str(round(val, 2))}')
        return max(child_vals, 
                          key=lambda child: child[1])[0]