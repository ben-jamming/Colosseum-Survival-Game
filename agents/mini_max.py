
class MiniMax():
    """
    General implementation of MiniMax algorithm
    
    a singleton class with one public method: get Action

    get_action: 
      - responsible for returning th minmaxed move given a state
      - params:
            - generate_children: a function that takes a state and returns a list of children states
                    - should be a list of states, where each state is a tuple of (board, player, adversary, max step)
            - utility: a function that takes a state and returns a utility value
                    - should return a return a float value for the utility of the state
            - is_terminal: a function that takes a state and returns a boolean
            - state: the current state of the game
            - state: [board, player, adversary, max step]
            - max_depth: the maximum depth to search

    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not MiniMax._instance:
            MiniMax._instance = super(MiniMax, cls).__new__(cls, *args, **kwargs)
        return MiniMax._instance
        
    def get_action(
        self,
        generate_children, 
        utility, 
        is_terminal,
        state, 
        max_depth=3
    ):
        
        def minimax(state, depth, maximizing_player):
            if depth == max_depth or is_terminal(state):
                return utility(state)
            if maximizing_player:
                return max(minimax(child, depth + 1, False) for child in generate_children(state))
            else:
                return min(minimax(child, depth + 1, True) for child in generate_children(state))
        return max(generate_children(state), key=lambda child: minimax(child, 0, False))
            
