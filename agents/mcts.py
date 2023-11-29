import time
import math
from turtle import undo
from typing import final
import numpy as np
import random
from agents.utils import perform_action, undo_last_action
from utils import *

# C is the exploration factor
C = 1.414

def uct(node):
    if node.visits == 0 or node.parent.visits == 0:
        return float('inf')
    #print(f"Calculating UCT: Wins: {node.wins}, Visits: {node.visits}, Parent Visits: {node.parent.visits}, UCT Value: {uct_value}")
    return node.wins / node.visits + C *  math.sqrt(math.log(node.parent.visits) / node.visits)

def random_child_expansion_policy(node):
  # Return a new move
  rand_val =random.randrange(len(node.unvisited_children_moves))
  return node.unvisited_children_moves.pop(rand_val)

def get_min_max_move(children, utility, state):
    # get the move with the best utility of its players turn
    # get the move with teh worst utility of its opponents turn
    def get_utility(move):
        perform_action(state, move)
        utility_score = utility(state)
        undo_last_action(state)
        return utility_score
    # sort the moves by their utility
    children.sort(key=get_utility)
    # pick randomly from the top k moves
    k = 1
    best_move = random.choice(children[:k])
    return best_move

def min_max_simulation(node, state, generate_children, utility, simulation_depth):
    children = generate_children(state)
    moves_simulated = 0
    for _ in range(simulation_depth):
        if len(children) == 0:
            break
        # get the move with the best utility of its players turn
        # get the move with teh worst utility of its opponents turn
        move = get_min_max_move(children, utility, state)
        perform_action(state, move)
        children = generate_children(state)
        moves_simulated += 1
    result = utility(state)
    for _ in range(moves_simulated):
        undo_last_action(state)
    return result

def random_simulation(node, state, generate_children, utility, simulation_depth):
  # Randomly simulate a bunch of games from the current expanded state
  # Assume that we are in the newly expanded node
  # Run the simulations on that node and its children up to the depth limit
  # Evaluate the utility of that state
  # Undo the simulated moves and restore the prior state
  #print(f"Starting simulation from node: {node}, Depth: {simulation_depth}")
  children = generate_children(state)
  moves_simulated = 0
  for _ in range(simulation_depth):
    if len(children) == 0:
      break
    move = random.choice(children)
    perform_action(state, move)
    children = generate_children(state)
    moves_simulated += 1

  result = utility(state)
  for _ in range(moves_simulated):
    undo_last_action(state)

  return result


class MCTS():
  """
  Monte Carlo Tree Search
        - Selection 
              - find the most promising node
              - if node is not fully expanded, expand it
              - what it means to be expanded:
                    - Each node has a set of children that are possible moves
                    - however, to start off with , these are all unvisited
                    - a child becomes visited we select it to run a simulation
                    - once all of childrens nodes are visited, the node is considered expanded
              - if it is expanded keep selecting until a leaf node is reached
              - Promising means that it has a high value and has not been explored much
              - formula: UCT = V + C * sqrt(ln(N) / n)
                - V = value of node
                - C = exploration parameter
                - N = number of times parent node has been visited
                - n = number of times child node has been visited
        - Expansion
              - If not expanded, select some amount of children to expand
        - Simulation
              - Run a simulation from the selected node to a certain depth
              - typically a random simulation where moves are chosen randomly
              - can also use a heuristic to guide the simulation
              - if we dont reach the end of the game, we need to evaluate the simulation or restart the simulation
              - we can evaluate using a utility function for the board
        - Backpropagation
              - use the result (win or loss) of the simulation to update the nodes
              - the nodes we need to update are the nodes that were visited during selection
              - update the value of the nodes by adding the result of the simulation to the value of the node
              - update the number of visits of the nodes by adding 1 to the number of visits of the node
  """

  def get_next_move(
      generate_children,
      utility,
      state,
      max_depth,
      simulation_depth,
      simulation_policy = random_simulation,
      child_expansion_policy = random_child_expansion_policy,
      time_limit = 2,
      memory_limit = 500, # in MB
      iterations = 1000,
      exploration_constant = 1.414,
      ):
    
    """
    Function conditions:
    - generate_children: should return a list of possible actions that can be taken from the current state
    - utilility: should return a value between -1 and 1, the utility function, 
        should be different for final game state and intermediate game states
        in the final game state, we want to return 1 if we won, -1 if we lost, and note use the heuristic
        in the intermediate game states, we want to return the heuristic value of the board

    - child_expansion_policy: needs to select and return a single child, and remove it from the list of unvisited children

    """
    
      #===========================================================================
    class Node():
      def __init__(self, parent=None, parent_move = None):
        self.parent = parent
        self.parent_move = parent_move # The move that led to this node
        self.children = [] # child nodes
        self.wins = 0
        self.visits = 0
        self.unvisited_children_moves = generate_children(state)
        #print(f"Created new node: {self.__str__()}")
      
      def __str__(self) -> str:
        return f"Node: Parent Move: {self.parent_move}, \
          Children: {self.children}, Wins: {self.wins}, \
            Visits: {self.visits}"

      def is_terminal_node(self):
        return len(self.children) == 0 and len(self.unvisited_children_moves) == 0
      
      def is_leaf_node(self):
        return len(self.children) == 0
      
      def is_fully_expanded(self):
        return len(self.unvisited_children_moves) == 0

      def update(self, result):
        self.wins += result
        self.visits += 1
        #print(f"Updating node {self}: Wins: {self.wins}, Visits: {self.visits}")
      
      def select_traversal_child(self):
        # if the node is the last node or if it has unvisited children, return None
        # pick child with best uct value
        chosen_child = self.children[0]
        max_uct = uct(chosen_child)
        for child in self.children:
          child_uct = uct(child)
          if child_uct > max_uct:
            max_uct = child_uct
            chosen_child = child
        return chosen_child
    
    prev_time = time.time()

    total_iterations = 0
    def resources_left():
        nonlocal total_iterations
        nonlocal iterations
        nonlocal time_limit
        nonlocal prev_time
        time_limit -= time.time() - prev_time
        prev_time = time.time()
        return total_iterations < iterations and time_limit > 0

    
    def selection(node: Node):
      # we want to keep going until we encounter a leaf node, ie a terminal node
      # or, until we encounter a node that has unvisited children
      # so we want keep searching as long as our current not has no unvisited children and is not a terminal node
      # We need a special case for the root because otherwise the loop will never enter since its technically terminal
      #print(f"The current node is {node.__str__()} and we need it to pick one of it's children")
      # try:
      #    print(node.unvisited_children_moves)

      # except AttributeError:
      #     print(f"Node {node.__str__()} has no unvisited children")
         
      # if node.parent is None:  # This is the root node
      #     if not node.unvisited_children_moves:
      #         node.unvisited_children_moves = generate_children(state)  # Initialize moves for root
      #     return node

      # Normal selection process for non-root nodes

      # print node.unvisited_children_moves

      while not node.is_leaf_node():
          if not node.is_fully_expanded():
              return node
          
          next_node = node.select_traversal_child()
          perform_action(state, next_node.parent_move)
          node = next_node
      #print(f"Selected a new node: {node}")
      return node

    def expansion(node: Node):
      # choose a move to a child node to expand
      # the expansion policy is responsible for removing the child from the list of unvisited children
      # when we pick this new move, we want to update the game state to reflect it
      # we then create the child node associated with this node
      if node.is_fully_expanded():
        return node

      move = child_expansion_policy(node)
      perform_action(state, move)
      new_child_node = Node(node, move)
      node.children.append(new_child_node)
      return new_child_node

    def backpropagation(node, result):
      # Were we supposed to also apply a move here?
      node.update(result)
      while node.parent != None:
        undo_last_action(state)
        node = node.parent
        node.update(result)

        # print(f"Selected node: {node.parent_move}, Visits: {node.visits}, Wins: {node.wins}")
      #print(f"Node to return is {node.__str__()}")
      return node
    
    def mcts(root):
      nonlocal total_iterations
      
      node = root
      
      while resources_left():
        total_iterations += 1
        # Select a new node from the current nodes children
        # Move to the selected node 
        # Expand the node according to the expansion policy
        # Run the simulations up to the specified depth and backpropagate
        # Restore the state to the expanded node
        new_node = selection(node)
         #print visits and wins and action


        new_child_node = expansion(new_node)
        result = simulation_policy(new_child_node, state, generate_children, utility, simulation_depth)
        node = backpropagation(new_child_node, result)
      
      #choose the bes
      for child in root.children:
         #print visits and wins
          perform_action(state, child.parent_move)
          #print(f"Child: {child.parent_move}, Visits: {child.visits}, Wins: {child.wins}")
          undo_last_action(state)

      # go through each child
      # pick the one with the best weighted score
      def weighted_score(node):

        return (node.wins / node.visits) * math.log(1 + node.visits)
      
      best_child = max(root.children, key=weighted_score)

      # if the best child isnet very good
      # we should just pick a move with the best utility score
      # this makes sense because mcts is look far ahead
      # but it doesnt consider how good the current state is
      # so it will pick a less simulated move that is technically better
      # but kills the player
      def get_utility(node):
        perform_action(state, node.parent_move)
        utility_score = utility(state)
        undo_last_action(state)
        return utility_score
      #print(f'weighted score of best child: {weighted_score(best_child)}')


      # if best_child.wins <= 0 or get_utility(best_child) <= -0.9:
      #   # print utility scores of each child
      #   for child in root.children:
      #     print(f"Child: {child.parent_move}, Utility: {get_utility(child)}")
      #   best_child = max(root.children, key=get_utility)
      

      


      return best_child.parent_move

    root = Node()
    choice = mcts(root)
    print(f'iterations: {total_iterations}')
    return choice

    


    
    
