class AlphaBeta:
    def get_action(generate_children, utility, state, max_depth):
        def minimax(state, depth, alpha, beta, max_player):
            # if max_player = True, then we are maximizing player
            if depth == max_depth or not generate_children(state):
                return utility(state) 
            
            cur_val = float('-inf') if max_player else float('inf')
            for child in generate_children(state):
                val = minimax(child, depth + 1, alpha, beta, not max_player)
                if max_player:
                    cur_val = max(cur_val, val)
                    alpha = max(alpha, cur_val)
                else: 
                    cur_val = min(cur_val, val)
                    beta = min(beta, cur_val)
                if beta <= alpha:
                    break
            
        
        return max(generate_children(state), 
                          key=lambda child: minimax(child, 0, float('-inf'), float('inf'), False), 
                          default=None)