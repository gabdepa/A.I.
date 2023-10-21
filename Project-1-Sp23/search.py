# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    stack = util.Stack()  # Initialize an empty stack
    visited = set()  # Initialize an empty set to keep track of visited states
    start_state = problem.getStartState()  # Get the start state from the problem
    # Push the start state and an empty actions list onto the stack
    stack.push((start_state, []))

    while not stack.isEmpty():  # Continue until the stack is empty
        # Pop the current state and actions so far from the stack
        current_state, actions = stack.pop()
        if current_state not in visited:  # If the current state has not been visited yet
            visited.add(current_state)  # Mark the current state as visited
            # If the current state is the goal state
            if problem.isGoalState(current_state):
                return actions  # Return the actions taken to reach the goal
            # Get the successors of the current state from the problem
            for successor, action, _ in problem.getSuccessors(current_state):
                if successor not in visited:  # If the successor has not been visited yet
                    new_actions = actions.copy()  # Copy the actions so far
                    # Add the action to reach the successor to the actions list
                    new_actions.append(action)
                    # Push the successor and updated actions onto the stack
                    stack.push((successor, new_actions))

    return []  # Return an empty list if no path to the goal is found


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    queue = util.Queue()  # Initialize an empty queue
    visited = set()  # Initialize an empty set to keep track of visited states
    start_state = problem.getStartState()  # Get the start state from the problem
    # Enqueue the start state and an empty actions list into the queue
    queue.push((start_state, []))

    while not queue.isEmpty():  # Continue until the queue is empty
        # Dequeue the current state and actions so far from the queue
        current_state, actions = queue.pop()
        if current_state not in visited:  # If the current state has not been visited yet
            visited.add(current_state)  # Mark the current state as visited
            # If the current state is the goal state
            if problem.isGoalState(current_state):
                return actions  # Return the actions taken to reach the goal
            # Get the successors of the current state from the problem
            for successor, action, _ in problem.getSuccessors(current_state):
                if successor not in visited:  # If the successor has not been visited yet
                    new_actions = actions.copy()  # Copy the actions so far
                    # Add the action to reach the successor to the actions list
                    new_actions.append(action)
                    # Enqueue the successor and updated actions into the queue
                    queue.push((successor, new_actions))
    return []  # Return an empty list if no path to the goal is found


def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    # Step 1: Initialization
    # Initialize a new instance of priority queue
    priority_queue = util.PriorityQueue()
    visited = {}  # Dictionary to keep track of visited states and their costs
    start_state = problem.getStartState()  # Get Start state
    # Enqueue the start state with a priority of 0
    priority_queue.push((start_state, []), 0)

    # Step 2: Exploration Loop
    while not priority_queue.isEmpty():
        # Dequeue the state with the lowest priority (cost)
        current_state, actions = priority_queue.pop()
        current_cost = problem.getCostOfActions(
            actions)  # Get the total cost so far
        if current_state in visited and visited[current_state] <= current_cost:
            continue  # Skip if this state has been visited with a lower or equal cost
        # Mark the current state as visited with its cost
        visited[current_state] = current_cost
        # If the current state is the goal state
        if problem.isGoalState(current_state):
            return actions  # Return the actions taken to reach the goal
        # Step 3: Enqueue Successors
        for successor, action, cost in problem.getSuccessors(current_state):
            if successor not in visited or visited[successor] > current_cost + cost:
                # Add the action to the actions list
                new_action = actions + [action]
                # Calculate the new priority (total cost)
                new_priority = current_cost + cost
                # Enqueue or update the successor
                priority_queue.update((successor, new_action), new_priority)

    return []  # Return an empty list if no path to the goal is found


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    # Step 1: Initialization
    # Create a function f(n) = g(n) + h(n) to use in the queue
    def priorityFunction(item):
        state, actions = item
        g = problem.getCostOfActions(actions)
        h = heuristic(state, problem)
        f = g + h
        return f
    
    # Initialize a new instance of priority queue with function defined above
    priority_queue = util.PriorityQueueWithFunction(priorityFunction)
    # Initialize an empty set to keep track of visited states
    visited = set()
    # Get Start state
    start_state = problem.getStartState()
    # Enqueue the start state
    priority_queue.push((start_state, []))

    # Step 2: Exploration Loop
    while not priority_queue.isEmpty():
        # Dequeue the state with the lowest priority (cost)
        current_state, actions = priority_queue.pop()
        if current_state in visited:  # Skip if this state has been visited
            continue
        # If the current state is the goal state
        if problem.isGoalState(current_state):
            return actions  # Return the actions taken to reach the goal
        visited.add(current_state)  # Add the current state to the visited set
        # Step 3: Enqueue Successors
        for successor, action, _ in problem.getSuccessors(current_state):
            # Add the action to the actions list
            new_actions = actions + [action]
            if successor not in visited:  # If the successor has not been visited yet
                # Enqueue the successor
                priority_queue.push((successor, new_actions))

    return []  # Return an empty list if no path to the goal is found


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
