from sqlite3 import Time
from tracemalloc import start
from .utils import *
import numpy as np
import signal
from memory_profiler import profile

def TimeLimitExceeded(Exception):
    pass

def handler(signum, frame):
    raise TimeLimitExceeded("Time limit exceeded")



class AlphaBeta:
    

    @profile
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
        iterations = 0

        def child_heuristic(child):
          player_pos = state['player']
          child_pos = child[0]
          adv_pos = state['adversary']
          distance_from_player = (player_pos[0] - child_pos[0])**2 + (player_pos[1] - child_pos[1])**2
          distance_to_adv = (adv_pos[0] - child_pos[0])**2 + (adv_pos[1] - child_pos[1])**2
          return distance_from_player * distance_to_adv  + distance_from_player
        
        def minimax(cur_state, depth, alpha, beta):
            nonlocal iterations

            # if max_player = True, then we are maximizing player
            current_time = time.time()
            if current_time - start_time > time_limit:
                return utility(cur_state,current_time, start_time)

            if depth == max_depth:
                return utility(cur_state,current_time, start_time)
            
            children = generate_children(cur_state, start_time, time_limit)
            children.sort(key=child_heuristic)
            children = children[:int(breadth_limit/depth)]
            
            if not children:
                return utility(cur_state, current_time, start_time)
            
            max_player = cur_state['is_player_turn']

            cur_val = float('-inf') if max_player else float('inf')

            

            for child in children:
                
                perform_action(cur_state, child)
                if current_time - start_time > time_limit:
                    break                
                val = minimax(cur_state, depth + 1, alpha, beta)
                if current_time - start_time > time_limit:
                    break
                undo_last_action(cur_state)
                iterations += 1

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
        children = generate_children(state, start_time, time_limit)
        # sort each child by its distance from the player
            
        # the sort function sorts in 
        children.sort(key=child_heuristic)
        children = children[:breadth_limit]
        max_child = children[0]
        max_val = float('-inf')

        try:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(int(time_limit))
            for child in children:
                perform_action(state, child)
                child_val = minimax(state, 1, float('-inf'), float('inf'))
                #child_val = float('-inf')
                undo_last_action(state)
                if child_val > max_val:
                    max_val = child_val
                    max_child = child

            signal.alarm(0)

        except TimeLimitExceeded:
            # Pick the best move available
            for child in children:
                perform_action(state, child)
                child_val = utility(state, start_time, time_limit)
                undo_last_action(state)
                if child_val > max_val:
                    max_val = child_val
                    max_child = child

        def get_utility(move):
            perform_action(state, move)
            val = utility(state, start_time, time_limit)
            undo_last_action(state)
            return val
        
        #print(f'chosen action: {max_child}, val: {max_val}, utility: {get_utility(max_child)}')
        if time.time() - start_time > time_limit:
            print("TIME LIMIT EXCEEDED")
            return max_child
        
        if max_val == -1:
            for child in children:
                perform_action(state, child)
                child_val = utility(state, start_time, time_limit)
                undo_last_action(state)
                if child_val > max_val:
                    max_val = child_val
                    max_child = child
            #print(f'chosen action: {max_child}, val: {max_val} utility: {get_utility(max_child)}')
        
        #print('iterations: ', iterations)


        

        return max_child