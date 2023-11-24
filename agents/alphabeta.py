import itertools as it
class AlphaBeta:
    def get_action(self, generate_children, utility, state, max_depth):
        def minimax(state, depth, alpha, beta, maximizing_player):
            # How this works:
            # - generate_children(state) returns a list of children states
            # - we want to go over each child, and update the max alpha and max val
            # - the accumulate fn goes over each child and keeps track of the max alpha and max val
            # - the accumulate fn returns a list of the max alpha and max val for each child up until that point
            # - we want to stop when the max alpha is greater than the min beta
            # - the takewhile fn goes through and stops when that condition is met
            # - we want to return the last value of the list, which is the max val hence [-1]
            # - the return value is a tuple of (max_alpha, max_val) so we want to return the max_val hence [1]
            # like wise for the minimizing player except the flip of max and min
            if depth == max_depth or not generate_children(state):
                return utility(state) 
            
            player = {
                True: {
                    'selector': max,
                    'initial': (alpha, float('-inf')),
                    'condition': lambda max_alpha, max_val: max_alpha < beta
                },
                False: {
                    'selector': min,
                    'initial': (beta, float('inf')),
                    'condition': lambda min_beta, min_val: min_beta > alpha
                }
            }
            
            def accumulate(accumulator, cur_child, selector):
                return (selector(accumulator[0], child_val := minimax(cur_child, depth + 1, accumulator[0], accumulator[1], not maximizing_player)), 
                        selector(accumulator[1], child_val))
            
            children_vals = it.accumulate(generate_children(state),
                                          lambda accumulator, cur_child: accumulate(accumulator, cur_child, player[maximizing_player]['selector']),
                                          initial=player[maximizing_player]['initial'])
            return it.takewhile(player[maximizing_player]['condition'], children_vals)[-1][1]

        best_child = max(generate_children(state), 
                          key=lambda child: minimax(child, 0, float('-inf'), float('inf'), False), 
                          default=None)
        return best_child