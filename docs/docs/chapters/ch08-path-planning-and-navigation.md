---
sidebar_position: 8
title: "Chapter 8: Path Planning and Navigation"
---

# Chapter 8: Path Planning and Navigation

## Learning Objectives

By the end of this chapter, students will be able to:
- Understand fundamental concepts of path planning and navigation for autonomous robots
- Implement classical path planning algorithms like A*, Dijkstra, and RRT
- Apply sampling-based motion planning techniques for high-dimensional spaces
- Design navigation systems that integrate perception, planning, and control
- Evaluate the performance of different path planning approaches in various scenarios
- Implement obstacle avoidance strategies for dynamic environments
- Develop multi-robot coordination and path planning systems
- Assess the trade-offs between optimality, completeness, and computational efficiency

## Theoretical Foundations

### Introduction to Path Planning

Path planning is a fundamental problem in robotics that involves finding a collision-free path from a start configuration to a goal configuration in an environment with obstacles. The path planning problem can be formally defined as follows: given a robot's configuration space (C-space), a start configuration (q_start), a goal configuration (q_goal), and a set of obstacles (O), find a continuous path in the free space (C_free = C - O) that connects q_start to q_goal.

The configuration space represents all possible configurations of the robot. For a 2D point robot, the configuration space is simply the 2D plane. For a robot arm with n joints, the configuration space is an n-dimensional space where each dimension corresponds to a joint angle. The dimensionality of the configuration space significantly impacts the complexity of path planning algorithms.

Path planning algorithms can be categorized based on several criteria:
- **Dimensionality**: 2D, 3D, or high-dimensional spaces
- **Environment type**: Static vs. dynamic environments
- **Solution properties**: Exact vs. approximate solutions
- **Optimality**: Optimal vs. feasible solutions
- **Computation approach**: Graph-based vs. sampling-based methods

### Configuration Space and Obstacle Representation

The configuration space (C-space) is a mathematical construct where each point represents a unique configuration of the robot. For a mobile robot moving in 2D space, the configuration space typically includes the robot's position (x, y) and orientation (θ), resulting in a 3D configuration space. For a manipulator with n joints, the configuration space is n-dimensional.

Obstacle representation is crucial for path planning. Common representations include:
- **Polygonal models**: Obstacles represented as polygons in 2D or polyhedra in 3D
- **Grid-based**: Environment discretized into a grid of cells
- **Point cloud**: Obstacles represented as sets of 3D points
- **Implicit surfaces**: Obstacles defined by mathematical functions

The Minkowski sum is often used to account for the robot's geometry when planning. The Minkowski sum of the robot and an obstacle represents the region in configuration space where the robot would collide with the obstacle.

### Path Planning Categories

Path planning algorithms can be broadly classified into two main categories:

**Combinatorial methods** discretize the configuration space into a graph structure. Examples include:
- Visibility graphs: Connect start, goal, and obstacle vertices
- Cell decomposition: Divide space into non-overlapping cells
- Roadmap methods: Create a network of paths in the free space

**Sampling-based methods** explore the configuration space by randomly sampling points and connecting them to form a graph. Examples include:
- Probabilistic Roadmaps (PRM): Pre-compute roadmap, then query for paths
- Rapidly-exploring Random Trees (RRT): Grow tree from start toward goal
- RRT*: Extension of RRT that provides asymptotic optimality

## Classical Path Planning Algorithms

### A* Algorithm Implementation

The A* algorithm is a popular graph-based path planning algorithm that guarantees optimal solutions under certain conditions. It uses a heuristic function to guide the search toward the goal efficiently.

```python
#!/usr/bin/env python3

import heapq
import numpy as np
from typing import List, Tuple, Optional

class Node:
    def __init__(self, position: Tuple[int, int], g_cost: float = 0, h_cost: float = 0, parent=None):
        self.position = position
        self.g_cost = g_cost  # Cost from start to current node
        self.h_cost = h_cost  # Heuristic cost from current node to goal
        self.f_cost = g_cost + h_cost  # Total cost
        self.parent = parent

    def __lt__(self, other):
        return self.f_cost < other.f_cost

    def __eq__(self, other):
        return self.position == other.position

class AStarPlanner:
    def __init__(self, grid: np.ndarray):
        self.grid = grid
        self.rows, self.cols = grid.shape
        # 8-directional movement
        self.directions = [(-1, -1), (-1, 0), (-1, 1),
                          (0, -1),           (0, 1),
                          (1, -1),  (1, 0),  (1, 1)]

    def heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate heuristic distance between two points (Euclidean)"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within grid bounds and not an obstacle"""
        x, y = pos
        return (0 <= x < self.rows and 0 <= y < self.cols and
                self.grid[x][y] == 0)  # 0 = free space, 1 = obstacle

    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring positions"""
        neighbors = []
        for dx, dy in self.directions:
            new_pos = (pos[0] + dx, pos[1] + dy)
            if self.is_valid(new_pos):
                # Calculate movement cost (diagonal = sqrt(2), straight = 1)
                cost = np.sqrt(2) if dx != 0 and dy != 0 else 1.0
                neighbors.append((new_pos, cost))
        return neighbors

    def plan_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Plan path using A* algorithm"""
        if not self.is_valid(start) or not self.is_valid(goal):
            return None

        # Initialize open and closed sets
        open_set = []
        closed_set = set()

        # Create start node
        start_node = Node(start, 0, self.heuristic(start, goal))
        heapq.heappush(open_set, start_node)

        # Keep track of visited nodes and their costs
        visited_costs = {start: start_node.f_cost}

        while open_set:
            current_node = heapq.heappop(open_set)

            # Check if we reached the goal
            if current_node.position == goal:
                return self.reconstruct_path(current_node)

            closed_set.add(current_node.position)

            # Explore neighbors
            for neighbor_pos, move_cost in self.get_neighbors(current_node.position):
                if neighbor_pos in closed_set:
                    continue

                # Calculate tentative g_cost
                tentative_g = current_node.g_cost + move_cost

                # Check if this path is better than previous ones
                if neighbor_pos in visited_costs and tentative_g >= visited_costs[neighbor_pos]:
                    continue

                # Create neighbor node
                h_cost = self.heuristic(neighbor_pos, goal)
                neighbor_node = Node(neighbor_pos, tentative_g, h_cost, current_node)

                heapq.heappush(open_set, neighbor_node)
                visited_costs[neighbor_pos] = neighbor_node.f_cost

        # No path found
        return None

    def reconstruct_path(self, node: Node) -> List[Tuple[int, int]]:
        """Reconstruct path from goal to start"""
        path = []
        current = node
        while current is not None:
            path.append(current.position)
            current = current.parent
        return path[::-1]  # Reverse to get start-to-goal path

# Example usage
if __name__ == "__main__":
    # Create a sample grid (0 = free space, 1 = obstacle)
    grid = np.array([
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

    planner = AStarPlanner(grid)
    start = (0, 0)
    goal = (9, 9)

    path = planner.plan_path(start, goal)

    if path:
        print(f"Path found with {len(path)} steps:")
        for i, pos in enumerate(path):
            print(f"Step {i}: {pos}")
    else:
        print("No path found")
```

### Dijkstra's Algorithm Implementation

Dijkstra's algorithm is another graph-based path planning method that guarantees optimal solutions but without using heuristics:

```python
#!/usr/bin/env python3

import heapq
import numpy as np
from typing import List, Tuple, Optional

class DijkstraPlanner:
    def __init__(self, grid: np.ndarray):
        self.grid = grid
        self.rows, self.cols = grid.shape
        # 4-directional movement (can be changed to 8-directional)
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within grid bounds and not an obstacle"""
        x, y = pos
        return (0 <= x < self.rows and 0 <= y < self.cols and
                self.grid[x][y] == 0)  # 0 = free space, 1 = obstacle

    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int, float]]:
        """Get valid neighboring positions with movement costs"""
        neighbors = []
        for dx, dy in self.directions:
            new_pos = (pos[0] + dx, pos[1] + dy)
            if self.is_valid(new_pos):
                # Movement cost is 1.0 for all directions (or can be weighted)
                cost = 1.0
                neighbors.append((new_pos[0], new_pos[1], cost))
        return neighbors

    def plan_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Plan path using Dijkstra's algorithm"""
        if not self.is_valid(start) or not self.is_valid(goal):
            return None

        # Initialize distances and previous nodes
        distances = {}
        previous = {}
        unvisited = []

        # Set all distances to infinity except start
        for x in range(self.rows):
            for y in range(self.cols):
                if self.grid[x][y] == 0:  # Only consider free space
                    distances[(x, y)] = float('inf')
                    previous[(x, y)] = None

        distances[start] = 0
        heapq.heappush(unvisited, (0, start))

        while unvisited:
            current_distance, current_pos = heapq.heappop(unvisited)

            # If we reached the goal, reconstruct path
            if current_pos == goal:
                return self.reconstruct_path(previous, start, goal)

            # If current distance is greater than stored distance, skip
            if current_distance > distances[current_pos]:
                continue

            # Check all neighbors
            for neighbor_x, neighbor_y, move_cost in self.get_neighbors(current_pos):
                neighbor_pos = (neighbor_x, neighbor_y)

                # Calculate new distance
                new_distance = distances[current_pos] + move_cost

                # If new path is shorter, update distance and previous
                if new_distance < distances[neighbor_pos]:
                    distances[neighbor_pos] = new_distance
                    previous[neighbor_pos] = current_pos
                    heapq.heappush(unvisited, (new_distance, neighbor_pos))

        # No path found
        return None

    def reconstruct_path(self, previous: dict, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from goal to start using previous nodes"""
        path = []
        current = goal

        while current is not None:
            path.append(current)
            current = previous[current]

        return path[::-1]  # Reverse to get start-to-goal path

# Example usage
if __name__ == "__main__":
    # Create a sample grid (0 = free space, 1 = obstacle)
    grid = np.array([
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

    planner = DijkstraPlanner(grid)
    start = (0, 0)
    goal = (9, 9)

    path = planner.plan_path(start, goal)

    if path:
        print(f"Dijkstra path found with {len(path)} steps:")
        for i, pos in enumerate(path):
            print(f"Step {i}: {pos}")
    else:
        print("No path found")
```

## Sampling-Based Motion Planning

### Rapidly-exploring Random Trees (RRT)

RRT is a sampling-based algorithm particularly effective for high-dimensional configuration spaces:

```python
#!/usr/bin/env python3

import numpy as np
import random
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

class RRTNode:
    def __init__(self, position: np.ndarray):
        self.position = position
        self.parent = None
        self.children = []

class RRT:
    def __init__(self, start: np.ndarray, goal: np.ndarray, bounds: Tuple[float, float, float, float],
                 obstacle_list: List[Tuple[float, float, float]], step_size: float = 0.5):
        """
        Initialize RRT planner
        :param start: Start position [x, y]
        :param goal: Goal position [x, y]
        :param bounds: Environment bounds (min_x, max_x, min_y, max_y)
        :param obstacle_list: List of obstacles [(x, y, radius), ...]
        :param step_size: Step size for extending the tree
        """
        self.start = RRTNode(np.array(start))
        self.goal = np.array(goal)
        self.bounds = bounds
        self.obstacles = obstacle_list
        self.step_size = step_size

        # Build tree
        self.tree = [self.start]
        self.goal_found = False

    def distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points"""
        return np.linalg.norm(pos1 - pos2)

    def is_collision(self, pos: np.ndarray) -> bool:
        """Check if position collides with any obstacle"""
        for obs_x, obs_y, obs_radius in self.obstacles:
            if self.distance(pos, np.array([obs_x, obs_y])) <= obs_radius:
                return True
        return False

    def is_valid_position(self, pos: np.ndarray) -> bool:
        """Check if position is within bounds and not colliding"""
        min_x, max_x, min_y, max_y = self.bounds
        if not (min_x <= pos[0] <= max_x and min_y <= pos[1] <= max_y):
            return False
        return not self.is_collision(pos)

    def find_nearest_node(self, pos: np.ndarray) -> RRTNode:
        """Find the nearest node in the tree to the given position"""
        nearest_node = self.tree[0]
        min_dist = self.distance(nearest_node.position, pos)

        for node in self.tree:
            dist = self.distance(node.position, pos)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node

        return nearest_node

    def extend_toward(self, from_node: RRTNode, target_pos: np.ndarray) -> Optional[RRTNode]:
        """Extend the tree toward the target position"""
        direction = target_pos - from_node.position
        dist = np.linalg.norm(direction)

        if dist <= self.step_size:
            new_pos = target_pos
        else:
            new_pos = from_node.position + (direction / dist) * self.step_size

        if self.is_valid_position(new_pos):
            new_node = RRTNode(new_pos)
            new_node.parent = from_node
            from_node.children.append(new_node)
            self.tree.append(new_node)

            # Check if we're close to the goal
            if self.distance(new_pos, self.goal) <= self.step_size:
                goal_node = RRTNode(self.goal)
                goal_node.parent = new_node
                new_node.children.append(goal_node)
                self.tree.append(goal_node)
                self.goal_found = True
                return goal_node

            return new_node

        return None

    def plan(self, max_iterations: int = 1000) -> Optional[List[np.ndarray]]:
        """Plan path using RRT algorithm"""
        for i in range(max_iterations):
            # Sample random position
            rand_x = random.uniform(self.bounds[0], self.bounds[1])
            rand_y = random.uniform(self.bounds[2], self.bounds[3])
            rand_pos = np.array([rand_x, rand_y])

            # Find nearest node in tree
            nearest_node = self.find_nearest_node(rand_pos)

            # Extend toward random position
            new_node = self.extend_toward(nearest_node, rand_pos)

            if self.goal_found:
                return self.extract_path()

        return None  # No path found

    def extract_path(self) -> List[np.ndarray]:
        """Extract path from goal to start"""
        if not self.goal_found:
            return []

        # Find goal node
        goal_node = None
        for node in self.tree:
            if self.distance(node.position, self.goal) <= self.step_size:
                goal_node = node
                break

        if goal_node is None:
            return []

        path = []
        current = goal_node
        while current is not None:
            path.append(current.position)
            current = current.parent

        return path[::-1]  # Reverse to get start-to-goal path

    def visualize(self, path: Optional[List[np.ndarray]] = None):
        """Visualize the RRT tree and path"""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot obstacles
        for obs_x, obs_y, obs_radius in self.obstacles:
            circle = plt.Circle((obs_x, obs_y), obs_radius, color='red', alpha=0.5)
            ax.add_patch(circle)

        # Plot tree edges
        for node in self.tree:
            if node.parent:
                ax.plot([node.position[0], node.parent.position[0]],
                       [node.position[1], node.parent.position[1]],
                       'b-', alpha=0.3, linewidth=0.5)

        # Plot nodes
        x_coords = [node.position[0] for node in self.tree]
        y_coords = [node.position[1] for node in self.tree]
        ax.scatter(x_coords, y_coords, c='blue', s=1, alpha=0.6)

        # Plot start and goal
        ax.plot(self.start.position[0], self.start.position[1], 'go', markersize=10, label='Start')
        ax.plot(self.goal[0], self.goal[1], 'ro', markersize=10, label='Goal')

        # Plot path if available
        if path:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            ax.plot(path_x, path_y, 'g-', linewidth=2, label='Path')

        ax.set_xlim(self.bounds[0], self.bounds[1])
        ax.set_ylim(self.bounds[2], self.bounds[3])
        ax.set_aspect('equal')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('RRT Path Planning')

        plt.show()

# Example usage
if __name__ == "__main__":
    # Define environment
    start = [1.0, 1.0]
    goal = [9.0, 9.0]
    bounds = (0, 10, 0, 10)

    # Define obstacles as (x, y, radius)
    obstacles = [
        (3, 3, 1.0),
        (5, 5, 1.0),
        (7, 2, 1.5),
        (2, 7, 1.2)
    ]

    # Create RRT planner
    rrt = RRT(start, goal, bounds, obstacles, step_size=0.5)

    # Plan path
    path = rrt.plan(max_iterations=1000)

    if path:
        print(f"Path found with {len(path)} waypoints")
        for i, pos in enumerate(path):
            print(f"Waypoint {i}: ({pos[0]:.2f}, {pos[1]:.2f})")
    else:
        print("No path found")

    # Visualize (uncomment to see the visualization)
    # rrt.visualize(path)
```

### RRT* Implementation

RRT* is an extension of RRT that provides asymptotic optimality:

```python
#!/usr/bin/env python3

import numpy as np
import random
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

class RRTStarNode:
    def __init__(self, position: np.ndarray):
        self.position = position
        self.parent = None
        self.children = []
        self.cost = 0.0  # Cost from start to this node

class RRTStar:
    def __init__(self, start: np.ndarray, goal: np.ndarray, bounds: Tuple[float, float, float, float],
                 obstacle_list: List[Tuple[float, float, float]], step_size: float = 0.5,
                 max_search_radius: float = 2.0):
        """
        Initialize RRT* planner
        :param start: Start position [x, y]
        :param goal: Goal position [x, y]
        :param bounds: Environment bounds (min_x, max_x, min_y, max_y)
        :param obstacle_list: List of obstacles [(x, y, radius), ...]
        :param step_size: Step size for extending the tree
        :param max_search_radius: Maximum radius to search for reconnection candidates
        """
        self.start = RRTStarNode(np.array(start))
        self.start.cost = 0.0
        self.goal = np.array(goal)
        self.bounds = bounds
        self.obstacles = obstacle_list
        self.step_size = step_size
        self.max_search_radius = max_search_radius

        # Build tree
        self.tree = [self.start]
        self.goal_found = False

    def distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points"""
        return np.linalg.norm(pos1 - pos2)

    def is_collision(self, pos1: np.ndarray, pos2: np.ndarray) -> bool:
        """Check if line segment between pos1 and pos2 collides with any obstacle"""
        # Simple implementation: check multiple points along the line
        steps = int(self.distance(pos1, pos2) / 0.1)  # Check every 0.1 units
        for i in range(steps + 1):
            t = i / steps if steps > 0 else 0
            point = pos1 + t * (pos2 - pos1)
            for obs_x, obs_y, obs_radius in self.obstacles:
                if self.distance(point, np.array([obs_x, obs_y])) <= obs_radius:
                    return True
        return False

    def is_valid_position(self, pos: np.ndarray) -> bool:
        """Check if position is within bounds and not colliding"""
        min_x, max_x, min_y, max_y = self.bounds
        if not (min_x <= pos[0] <= max_x and min_y <= pos[1] <= max_y):
            return False
        return not self.is_collision(pos, pos)  # Check for point collision

    def find_nearest_node(self, pos: np.ndarray) -> RRTStarNode:
        """Find the nearest node in the tree to the given position"""
        nearest_node = self.tree[0]
        min_dist = self.distance(nearest_node.position, pos)

        for node in self.tree:
            dist = self.distance(node.position, pos)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node

        return nearest_node

    def find_near_nodes(self, pos: np.ndarray, radius: float) -> List[RRTStarNode]:
        """Find all nodes within a certain radius of the given position"""
        near_nodes = []
        for node in self.tree:
            if self.distance(node.position, pos) <= radius:
                near_nodes.append(node)
        return near_nodes

    def choose_parent(self, near_nodes: List[RRTStarNode], new_pos: np.ndarray) -> Tuple[Optional[RRTStarNode], float]:
        """Choose parent for new node based on minimum cost"""
        min_cost = float('inf')
        best_parent = None

        for node in near_nodes:
            if not self.is_collision(node.position, new_pos):
                cost = node.cost + self.distance(node.position, new_pos)
                if cost < min_cost:
                    min_cost = cost
                    best_parent = node

        return best_parent, min_cost

    def rewire(self, new_node: RRTStarNode, near_nodes: List[RRTStarNode]):
        """Rewire the tree to improve path cost"""
        for node in near_nodes:
            if node != new_node.parent:  # Don't rewire to the new node's parent
                new_cost = new_node.cost + self.distance(new_node.position, node.position)
                if new_cost < node.cost and not self.is_collision(new_node.position, node.position):
                    # Update parent
                    if node.parent:
                        node.parent.children.remove(node)
                    node.parent = new_node
                    new_node.children.append(node)
                    node.cost = new_cost

    def extend_toward(self, from_node: RRTStarNode, target_pos: np.ndarray) -> Optional[RRTStarNode]:
        """Extend the tree toward the target position"""
        direction = target_pos - from_node.position
        dist = np.linalg.norm(direction)

        if dist <= self.step_size:
            new_pos = target_pos
        else:
            new_pos = from_node.position + (direction / dist) * self.step_size

        if self.is_valid_position(new_pos) and not self.is_collision(from_node.position, new_pos):
            # Find nearby nodes within search radius
            near_nodes = self.find_near_nodes(new_pos, self.max_search_radius)

            # Choose parent based on minimum cost
            parent, cost = self.choose_parent(near_nodes, new_pos)

            if parent is not None:
                new_node = RRTStarNode(new_pos)
                new_node.parent = parent
                new_node.cost = cost
                parent.children.append(new_node)
                self.tree.append(new_node)

                # Rewire the tree
                self.rewire(new_node, near_nodes)

                # Check if we're close to the goal
                if self.distance(new_pos, self.goal) <= self.step_size:
                    goal_node = RRTStarNode(self.goal)
                    goal_node.parent = new_node
                    goal_node.cost = new_node.cost + self.distance(new_node.position, self.goal)
                    new_node.children.append(goal_node)
                    self.tree.append(goal_node)
                    self.goal_found = True
                    return goal_node

                return new_node

        return None

    def plan(self, max_iterations: int = 2000) -> Optional[List[np.ndarray]]:
        """Plan path using RRT* algorithm"""
        for i in range(max_iterations):
            # Sample random position
            rand_x = random.uniform(self.bounds[0], self.bounds[1])
            rand_y = random.uniform(self.bounds[2], self.bounds[3])
            rand_pos = np.array([rand_x, rand_y])

            # Find nearest node in tree
            nearest_node = self.find_nearest_node(rand_pos)

            # Extend toward random position
            new_node = self.extend_toward(nearest_node, rand_pos)

        # After planning, try to connect to goal if not already connected
        if not self.goal_found:
            # Try to connect from the closest node to goal
            closest_node = min(self.tree, key=lambda n: self.distance(n.position, self.goal))
            if not self.is_collision(closest_node.position, self.goal):
                goal_node = RRTStarNode(self.goal)
                goal_node.parent = closest_node
                goal_node.cost = closest_node.cost + self.distance(closest_node.position, self.goal)
                closest_node.children.append(goal_node)
                self.tree.append(goal_node)
                self.goal_found = True

        if self.goal_found:
            return self.extract_path()

        return None  # No path found

    def extract_path(self) -> List[np.ndarray]:
        """Extract path from goal to start"""
        if not self.goal_found:
            return []

        # Find goal node
        goal_node = None
        for node in self.tree:
            if self.distance(node.position, self.goal) <= self.step_size:
                goal_node = node
                break

        if goal_node is None:
            return []

        path = []
        current = goal_node
        while current is not None:
            path.append(current.position)
            current = current.parent

        return path[::-1]  # Reverse to get start-to-goal path

    def get_path_cost(self) -> float:
        """Get the cost of the path to the goal"""
        if not self.goal_found:
            return float('inf')

        for node in self.tree:
            if self.distance(node.position, self.goal) <= self.step_size:
                return node.cost

        return float('inf')

# Example usage
if __name__ == "__main__":
    # Define environment
    start = [1.0, 1.0]
    goal = [9.0, 9.0]
    bounds = (0, 10, 0, 10)

    # Define obstacles as (x, y, radius)
    obstacles = [
        (3, 3, 1.0),
        (5, 5, 1.0),
        (7, 2, 1.5),
        (2, 7, 1.2)
    ]

    # Create RRT* planner
    rrt_star = RRTStar(start, goal, bounds, obstacles, step_size=0.5, max_search_radius=2.0)

    # Plan path
    path = rrt_star.plan(max_iterations=1000)

    if path:
        cost = rrt_star.get_path_cost()
        print(f"Path found with {len(path)} waypoints, total cost: {cost:.2f}")
        for i, pos in enumerate(path):
            print(f"Waypoint {i}: ({pos[0]:.2f}, {pos[1]:.2f})")
    else:
        print("No path found")
```

## Navigation Systems

### Global and Local Path Planning Integration

A complete navigation system typically combines global path planning with local obstacle avoidance:

```python
#!/usr/bin/env python3

import numpy as np
import heapq
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

class NavigationSystem:
    def __init__(self, grid: np.ndarray, robot_radius: float = 0.5):
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.robot_radius = robot_radius

        # Global planner (A*)
        self.global_planner = AStarPlanner(grid)

        # Local planner (Vector Field Histogram or similar)
        self.local_planner = LocalPlanner(robot_radius)

        # Current robot position and goal
        self.current_pos = None
        self.goal_pos = None

        # Global path (computed by global planner)
        self.global_path = []
        self.path_index = 0

    def set_start_goal(self, start: Tuple[int, int], goal: Tuple[int, int]):
        """Set start and goal positions"""
        self.current_pos = start
        self.goal_pos = goal

    def plan_global_path(self) -> bool:
        """Plan global path using A*"""
        path = self.global_planner.plan_path(self.current_pos, self.goal_pos)
        if path:
            self.global_path = path
            self.path_index = 0
            return True
        return False

    def navigate(self) -> Optional[Tuple[float, float]]:
        """Navigate toward goal, handling local obstacles"""
        if self.path_index >= len(self.global_path):
            return None  # Reached goal or no path

        # Get next waypoint from global path
        next_waypoint = self.global_path[self.path_index]

        # Check if we've reached the current waypoint
        if self.distance(self.current_pos, next_waypoint) <= 1.0:
            self.path_index += 1
            if self.path_index >= len(self.global_path):
                return None  # Reached goal
            next_waypoint = self.global_path[self.path_index]

        # Convert grid coordinates to continuous coordinates
        target_x = float(next_waypoint[1])
        target_y = float(next_waypoint[0])

        current_x = float(self.current_pos[1])
        current_y = float(self.current_pos[0])

        # Use local planner to navigate toward waypoint while avoiding obstacles
        next_pos = self.local_planner.plan_step(
            (current_x, current_y),
            (target_x, target_y),
            self.grid
        )

        return next_pos

    def distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

class LocalPlanner:
    def __init__(self, robot_radius: float = 0.5):
        self.robot_radius = robot_radius
        self.safe_distance = robot_radius * 2.0  # Minimum distance to obstacles

    def plan_step(self, current_pos: Tuple[float, float], target_pos: Tuple[float, float],
                  grid: np.ndarray) -> Tuple[float, float]:
        """Plan next step considering local obstacles"""
        current_x, current_y = current_pos
        target_x, target_y = target_pos

        # Calculate desired direction toward target
        dx = target_x - current_x
        dy = target_y - current_y
        distance_to_target = np.sqrt(dx**2 + dy**2)

        if distance_to_target < 0.1:  # Very close to target
            return target_pos

        # Normalize direction
        desired_direction = np.array([dx, dy]) / distance_to_target

        # Check for obstacles in the path
        obstacle_force = self.calculate_obstacle_force(current_pos, grid)

        # Combine desired direction with obstacle avoidance
        if np.linalg.norm(obstacle_force) > 0.1:
            # If there are significant obstacles, modify the direction
            combined_direction = desired_direction + 0.5 * obstacle_force
            combined_direction = combined_direction / np.linalg.norm(combined_direction)
        else:
            # No significant obstacles, go straight toward target
            combined_direction = desired_direction

        # Move one step in the calculated direction
        step_size = 0.5  # Adjust based on your requirements
        new_x = current_x + step_size * combined_direction[0]
        new_y = current_y + step_size * combined_direction[1]

        return (new_x, new_y)

    def calculate_obstacle_force(self, pos: Tuple[float, float], grid: np.ndarray) -> np.ndarray:
        """Calculate repulsive force from nearby obstacles"""
        x, y = pos
        force = np.array([0.0, 0.0])

        # Search in a local neighborhood
        search_radius = int(self.safe_distance * 2)

        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                nx, ny = int(x + dx), int(y + dy)

                # Check bounds
                if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                    if grid[nx, ny] == 1:  # Obstacle detected
                        # Calculate direction away from obstacle
                        obs_vec = np.array([x - nx, y - ny])
                        dist = np.linalg.norm(obs_vec)

                        if dist > 0:
                            # Normalize and apply repulsive force
                            obs_vec = obs_vec / dist
                            # Force magnitude decreases with distance
                            force_magnitude = 1.0 / (dist + 0.1)**2
                            force += force_magnitude * obs_vec

        # Normalize force
        force_norm = np.linalg.norm(force)
        if force_norm > 0:
            force = force / force_norm * min(force_norm, 1.0)

        return force

class DynamicWindowApproach:
    """Implementation of Dynamic Window Approach for local path planning"""

    def __init__(self, robot_params: dict):
        """
        Initialize DWA planner
        :param robot_params: Dictionary with robot parameters
        """
        self.max_speed = robot_params.get('max_speed', 1.0)
        self.min_speed = robot_params.get('min_speed', -0.5)
        self.max_yaw_rate = robot_params.get('max_yaw_rate', 40.0 * np.pi / 180.0)
        self.max_accel = robot_params.get('max_accel', 0.5)
        self.max_delta_yaw_rate = robot_params.get('max_delta_yaw_rate', 40.0 * np.pi / 180.0)
        self.dt = robot_params.get('dt', 0.1)
        self.predict_time = robot_params.get('predict_time', 3.0)
        self.to_goal_cost_gain = robot_params.get('to_goal_cost_gain', 0.15)
        self.speed_cost_gain = robot_params.get('speed_cost_gain', 1.0)
        self.obstacle_cost_gain = robot_params.get('obstacle_cost_gain', 1.0)
        self.robot_radius = robot_params.get('robot_radius', 1.0)

    def plan(self, state: np.ndarray, goal: np.ndarray, obstacles: np.ndarray) -> np.ndarray:
        """
        Plan next control command using DWA
        :param state: Current state [x, y, yaw, v, omega]
        :param goal: Goal position [x, y]
        :param obstacles: Obstacle positions [[x1, y1], [x2, y2], ...]
        :return: Next state after applying control
        """
        # Generate dynamic window
        window = self.calc_dynamic_window(state)

        # Evaluate trajectories
        best_traj = None
        best_score = float('-inf')

        for v in np.arange(window[0], window[1], self.max_accel * self.dt / 5.0):
            for yaw_rate in np.arange(window[2], window[3], self.max_delta_yaw_rate * self.dt / 5.0):
                # Simulate trajectory
                traj = self.predict_trajectory(state, v, yaw_rate)

                # Calculate costs
                to_goal_cost = self.calc_to_goal_cost(traj, goal)
                speed_cost = self.calc_speed_cost(traj)
                obstacle_cost = self.calc_obstacle_cost(traj, obstacles)

                # Calculate total score
                final_score = (self.to_goal_cost_gain * to_goal_cost +
                              self.speed_cost_gain * speed_cost -
                              self.obstacle_cost_gain * obstacle_cost)

                if final_score > best_score:
                    best_score = final_score
                    best_traj = [v, yaw_rate]

        if best_traj is None:
            best_traj = [0.0, 0.0]  # Stop if no valid trajectory found

        # Update state based on best trajectory
        state[2] += best_traj[1] * self.dt  # yaw update
        state[0] += best_traj[0] * np.cos(state[2]) * self.dt  # x update
        state[1] += best_traj[0] * np.sin(state[2]) * self.dt  # y update
        state[3] = best_traj[0]  # v update
        state[4] = best_traj[1]  # omega update

        return state

    def calc_dynamic_window(self, state: np.ndarray) -> np.ndarray:
        """Calculate dynamic window"""
        vs = np.array([self.min_speed, self.max_speed,
                      -self.max_yaw_rate, self.max_yaw_rate])

        vd = np.array([state[3] - self.max_accel * self.dt,
                      state[3] + self.max_accel * self.dt,
                      state[4] - self.max_delta_yaw_rate * self.dt,
                      state[4] + self.max_delta_yaw_rate * self.dt])

        # Return the intersection of state and velocity windows
        return np.array([max(vs[0], vd[0]), min(vs[1], vd[1]),
                        max(vs[2], vd[2]), min(vs[3], vd[3])])

    def predict_trajectory(self, state: np.ndarray, v: float, yaw_rate: float) -> np.ndarray:
        """Predict trajectory with given velocity and yaw rate"""
        traj = np.array(state)
        time = 0
        while time <= self.predict_time:
            state = self.motion(state, [v, yaw_rate])
            traj = np.vstack((traj, state))
            time += self.dt

        return traj

    def motion(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """Motion model"""
        state[0] += control[0] * np.cos(state[2]) * self.dt
        state[1] += control[0] * np.sin(state[2]) * self.dt
        state[2] += control[1] * self.dt
        state[3] = control[0]
        state[4] = control[1]

        return state

    def calc_to_goal_cost(self, traj: np.ndarray, goal: np.ndarray) -> float:
        """Calculate cost to goal"""
        dx = goal[0] - traj[-1, 0]
        dy = goal[1] - traj[-1, 1]
        error_angle = np.arctan2(dy, dx)
        cost_angle = error_angle - traj[-1, 2]
        cost = abs(np.arctan2(np.sin(cost_angle), np.cos(cost_angle)))

        return cost

    def calc_speed_cost(self, traj: np.ndarray) -> float:
        """Calculate speed cost"""
        return abs(self.max_speed - traj[-1, 3])

    def calc_obstacle_cost(self, traj: np.ndarray, obstacles: np.ndarray) -> float:
        """Calculate obstacle cost"""
        min_dist = float('inf')
        for i in range(len(traj)):
            for j in range(len(obstacles)):
                dist = np.sqrt((traj[i, 0] - obstacles[j, 0])**2 +
                              (traj[i, 1] - obstacles[j, 1])**2)
                if dist <= self.robot_radius:
                    return float('inf')  # Collision
                if dist < min_dist:
                    min_dist = dist

        return 1.0 / min_dist if min_dist != float('inf') else 0

# Example usage
if __name__ == "__main__":
    # Create a sample grid
    grid = np.array([
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

    # Initialize navigation system
    nav_system = NavigationSystem(grid, robot_radius=0.5)
    nav_system.set_start_goal((0, 0), (9, 9))

    # Plan global path
    if nav_system.plan_global_path():
        print("Global path planned successfully")

        # Simulate navigation
        current_pos = nav_system.current_pos
        path_taken = [current_pos]

        for step in range(50):  # Maximum steps
            next_pos = nav_system.navigate()
            if next_pos is None:
                print("Reached goal!")
                break
            path_taken.append((int(next_pos[1]), int(next_pos[0])))  # Convert back to grid coordinates
            nav_system.current_pos = (int(next_pos[1]), int(next_pos[0]))

        print(f"Navigation completed. Path length: {len(path_taken)} steps")
    else:
        print("Failed to plan global path")
```

## Obstacle Avoidance Strategies

### Vector Field Histogram Implementation

Vector Field Histogram (VFH) is a local navigation method that uses a histogram of obstacle directions:

```python
#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

class VectorFieldHistogram:
    def __init__(self, robot_radius: float = 0.5, sensor_range: float = 3.0,
                 num_sectors: int = 72, safety_threshold: float = 0.5):
        """
        Initialize VFH
        :param robot_radius: Radius of the robot
        :param sensor_range: Maximum sensor range
        :param num_sectors: Number of angular sectors for histogram
        :param safety_threshold: Minimum safe distance
        """
        self.robot_radius = robot_radius
        self.sensor_range = sensor_range
        self.num_sectors = num_sectors
        self.safety_threshold = safety_threshold
        self.angle_resolution = 2 * np.pi / num_sectors

    def build_histogram(self, robot_pos: Tuple[float, float],
                       obstacles: List[Tuple[float, float, float]]) -> np.ndarray:
        """
        Build polar histogram from obstacle data
        :param robot_pos: Current robot position (x, y)
        :param obstacles: List of obstacles [(x, y, radius), ...]
        :return: Polar histogram
        """
        histogram = np.zeros(self.num_sectors)

        robot_x, robot_y = robot_pos

        for obs_x, obs_y, obs_radius in obstacles:
            # Calculate distance and angle to obstacle
            dx = obs_x - robot_x
            dy = obs_y - robot_y
            distance = np.sqrt(dx**2 + dy**2)

            if distance <= self.sensor_range:
                # Calculate angle (in radians)
                angle = np.arctan2(dy, dx)
                if angle < 0:
                    angle += 2 * np.pi

                # Calculate the angle occupied by this obstacle
                # The angle subtended by the obstacle at the robot
                subtended_angle = 2 * np.arcsin((self.robot_radius + obs_radius) / max(distance, 0.1))

                # Find the sector(s) this obstacle occupies
                center_sector = int(angle / self.angle_resolution)
                sectors_occupied = int(subtended_angle / self.angle_resolution) + 1

                # Mark sectors as occupied
                for i in range(-sectors_occupied//2, sectors_occupied//2 + 1):
                    sector_idx = (center_sector + i) % self.num_sectors
                    # Weight by inverse of distance (closer obstacles get higher values)
                    histogram[sector_idx] = max(histogram[sector_idx],
                                              (self.sensor_range - distance) / self.sensor_range)

        return histogram

    def select_direction(self, histogram: np.ndarray, target_angle: float) -> float:
        """
        Select the best direction based on histogram and target
        :param histogram: Polar histogram
        :param target_angle: Desired direction toward goal
        :return: Selected direction angle
        """
        # Convert target angle to sector
        target_sector = int(target_angle / self.angle_resolution) % self.num_sectors

        # Find admissible directions (sectors with low obstacle density)
        threshold = 0.3  # Adjust based on desired safety level
        admissible_sectors = np.where(histogram < threshold)[0]

        if len(admissible_sectors) == 0:
            # No safe directions - return original target (risky)
            return target_angle

        # Find the admissible sector closest to target direction
        best_sector = admissible_sectors[0]
        min_diff = abs(best_sector - target_sector)

        for sector in admissible_sectors:
            diff = min(abs(sector - target_sector),
                      self.num_sectors - abs(sector - target_sector))
            if diff < min_diff:
                min_diff = diff
                best_sector = sector

        return best_sector * self.angle_resolution

    def plan_step(self, robot_pos: Tuple[float, float],
                  target_pos: Tuple[float, float],
                  obstacles: List[Tuple[float, float, float]],
                  step_size: float = 0.5) -> Tuple[float, float]:
        """
        Plan next step using VFH
        :param robot_pos: Current robot position
        :param target_pos: Target position
        :param obstacles: List of obstacles
        :param step_size: Step size for movement
        :return: Next position
        """
        # Calculate desired direction to target
        dx = target_pos[0] - robot_pos[0]
        dy = target_pos[1] - robot_pos[1]
        target_angle = np.arctan2(dy, dx)

        # Build histogram
        histogram = self.build_histogram(robot_pos, obstacles)

        # Select safe direction
        selected_angle = self.select_direction(histogram, target_angle)

        # Move in selected direction
        new_x = robot_pos[0] + step_size * np.cos(selected_angle)
        new_y = robot_pos[1] + step_size * np.sin(selected_angle)

        return (new_x, new_y)

    def visualize(self, robot_pos: Tuple[float, float],
                  obstacles: List[Tuple[float, float, float]],
                  target_pos: Tuple[float, float] = None):
        """Visualize the VFH"""
        histogram = self.build_histogram(robot_pos, obstacles)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot obstacles and robot
        ax1.set_xlim(-5, 5)
        ax1.set_ylim(-5, 5)
        ax1.set_aspect('equal')

        # Plot robot
        robot_circle = plt.Circle(robot_pos, self.robot_radius, color='blue', alpha=0.5, label='Robot')
        ax1.add_patch(robot_circle)

        # Plot obstacles
        for obs_x, obs_y, obs_radius in obstacles:
            obs_circle = plt.Circle((obs_x, obs_y), obs_radius, color='red', alpha=0.5, label='Obstacle' if obs_x == obstacles[0][0] else "")
            ax1.add_patch(obs_circle)

        # Plot target if provided
        if target_pos:
            ax1.plot(target_pos[0], target_pos[1], 'go', markersize=10, label='Target')

        # Draw sensor range
        sensor_circle = plt.Circle(robot_pos, self.sensor_range, color='gray', alpha=0.2, fill=False, label='Sensor Range')
        ax1.add_patch(sensor_circle)

        ax1.set_title('Obstacle Environment')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot histogram
        angles = np.linspace(0, 2*np.pi, self.num_sectors, endpoint=False)
        ax2.plot(angles, histogram, 'b-', linewidth=2)
        ax2.fill_between(angles, histogram, alpha=0.3)
        ax2.set_theta_zero_location('N')
        ax2.set_theta_direction(-1)
        ax2.set_title('Polar Histogram')

        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize VFH
    vfh = VectorFieldHistogram(robot_radius=0.5, sensor_range=3.0)

    # Define robot position, target, and obstacles
    robot_pos = (0.0, 0.0)
    target_pos = (4.0, 4.0)
    obstacles = [
        (1.0, 1.0, 0.5),
        (2.0, 2.0, 0.7),
        (3.0, 1.0, 0.4),
        (1.5, 3.0, 0.6)
    ]

    # Plan a single step
    next_pos = vfh.plan_step(robot_pos, target_pos, obstacles)
    print(f"Next position: {next_pos}")

    # Visualize (uncomment to see visualization)
    # vfh.visualize(robot_pos, obstacles, target_pos)
```

## Multi-Robot Path Planning

### Conflict-Based Search (CBS) for Multi-Agent Path Finding

```python
#!/usr/bin/env python3

import numpy as np
from typing import List, Tuple, Dict, Set
import heapq
from dataclasses import dataclass
import copy

@dataclass
class Constraint:
    agent_id: int
    location: Tuple[int, int]
    timestep: int

@dataclass
class PathNode:
    location: Tuple[int, int]
    timestep: int
    f_score: int
    g_score: int
    parent: 'PathNode'

    def __lt__(self, other):
        return self.f_score < other.f_score

class SingleAgentPlanner:
    """Path planner for a single agent using A* with constraints"""

    def __init__(self, grid: np.ndarray):
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.directions = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]  # Wait, N, S, W, E

    def heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Manhattan distance heuristic"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int],
                  constraints: List[Constraint], agent_id: int) -> List[Tuple[int, int]]:
        """Find path using A* with constraints"""
        open_set = []
        closed_set = set()

        start_node = PathNode(start, 0, 0, 0, None)
        heapq.heappush(open_set, start_node)

        while open_set:
            current = heapq.heappop(open_set)

            # Check if we reached the goal
            if current.location == goal:
                # Extract path
                path = []
                node = current
                while node:
                    path.append(node.location)
                    node = node.parent
                return path[::-1]

            # Check constraints
            if self.is_constrained(current.location, current.timestep, constraints, agent_id):
                continue

            # Add to closed set
            closed_set.add((current.location, current.timestep))

            # Explore neighbors
            for dx, dy in self.directions:
                next_pos = (current.location[0] + dx, current.location[1] + dy)
                next_timestep = current.timestep + 1

                # Check if move is valid
                if (0 <= next_pos[0] < self.rows and
                    0 <= next_pos[1] < self.cols and
                    self.grid[next_pos[0]][next_pos[1]] == 0 and  # No obstacle
                    (next_pos, next_timestep) not in closed_set and
                    not self.is_constrained(next_pos, next_timestep, constraints, agent_id)):

                    g_score = current.g_score + 1
                    h_score = self.heuristic(next_pos, goal)
                    f_score = g_score + h_score

                    next_node = PathNode(next_pos, next_timestep, f_score, g_score, current)
                    heapq.heappush(open_set, next_node)

        return []  # No path found

    def is_constrained(self, location: Tuple[int, int], timestep: int,
                      constraints: List[Constraint], agent_id: int) -> bool:
        """Check if location is constrained for this agent at this timestep"""
        for constraint in constraints:
            if (constraint.agent_id == agent_id and
                constraint.location == location and
                constraint.timestep == timestep):
                return True
        return False

class CBSNode:
    """Node for CBS tree"""

    def __init__(self, paths: Dict[int, List[Tuple[int, int]]], constraints: List[Constraint],
                 cost: int, num_conflicts: int):
        self.paths = paths
        self.constraints = constraints
        self.cost = cost
        self.num_conflicts = num_conflicts

    def __lt__(self, other):
        return self.num_conflicts < other.num_conflicts

class MultiAgentPathPlanner:
    """Multi-agent path planner using Conflict-Based Search (CBS)"""

    def __init__(self, grid: np.ndarray):
        self.grid = grid
        self.planner = SingleAgentPlanner(grid)

    def find_conflicts(self, paths: Dict[int, List[Tuple[int, int]]]) -> List[Tuple[int, int, Tuple[int, int], int]]:
        """Find conflicts between paths"""
        conflicts = []
        agents = list(paths.keys())

        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                agent1, agent2 = agents[i], agents[j]
                path1, path2 = paths[agent1], paths[agent2]

                # Make paths same length by extending with the last position
                max_len = max(len(path1), len(path2))
                path1_extended = path1 + [path1[-1]] * (max_len - len(path1)) if path1 else []
                path2_extended = path2 + [path2[-1]] * (max_len - len(path2)) if path2 else []

                # Check for vertex conflicts (same location at same time)
                for t in range(max_len):
                    if path1_extended[t] == path2_extended[t]:
                        conflicts.append((agent1, agent2, path1_extended[t], t))

                # Check for edge conflicts (swapping locations at same time)
                for t in range(1, max_len):
                    if (path1_extended[t-1] == path2_extended[t] and
                        path1_extended[t] == path2_extended[t-1]):
                        conflicts.append((agent1, agent2, (path1_extended[t-1], path1_extended[t]), t))

        return conflicts

    def plan_paths(self, starts: List[Tuple[int, int]], goals: List[Tuple[int, int]]) -> Dict[int, List[Tuple[int, int]]]:
        """Plan paths for multiple agents using CBS"""
        # Initialize with empty constraints
        initial_constraints = []

        # Find initial paths for all agents
        initial_paths = {}
        initial_cost = 0

        for agent_id, (start, goal) in enumerate(zip(starts, goals)):
            path = self.planner.find_path(start, goal, initial_constraints, agent_id)
            if not path:
                return {}  # No solution possible
            initial_paths[agent_id] = path
            initial_cost += len(path) - 1  # Path length as cost

        # Check for conflicts in initial solution
        conflicts = self.find_conflicts(initial_paths)

        if not conflicts:
            return initial_paths  # Solution found

        # Use CBS to resolve conflicts
        open_list = []
        root_node = CBSNode(initial_paths, initial_constraints, initial_cost, len(conflicts))
        heapq.heappush(open_list, root_node)

        while open_list:
            node = heapq.heappop(open_list)

            # Get first conflict
            conflict = self.find_conflicts(node.paths)[0]
            agent1_id, agent2_id, location, timestep = conflict

            # Create two child nodes with different constraints
            for constrained_agent in [agent1_id, agent2_id]:
                new_constraints = copy.deepcopy(node.constraints)
                new_constraint = Constraint(constrained_agent, location, timestep)
                new_constraints.append(new_constraint)

                # Recalculate path for the constrained agent
                new_paths = copy.deepcopy(node.paths)
                start = starts[constrained_agent]
                goal = goals[constrained_agent]
                new_path = self.planner.find_path(start, goal, new_constraints, constrained_agent)

                if new_path:  # Path exists with new constraint
                    new_paths[constrained_agent] = new_path

                    # Calculate new cost
                    new_cost = sum(len(path) - 1 for path in new_paths.values())

                    # Check for remaining conflicts
                    remaining_conflicts = self.find_conflicts(new_paths)

                    if not remaining_conflicts:
                        return new_paths  # Solution found

                    child_node = CBSNode(new_paths, new_constraints, new_cost, len(remaining_conflicts))
                    heapq.heappush(open_list, child_node)

        return {}  # No solution found

# Example usage
if __name__ == "__main__":
    # Create a sample grid
    grid = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0]
    ])

    # Initialize multi-agent planner
    multi_planner = MultiAgentPathPlanner(grid)

    # Define starts and goals for multiple agents
    starts = [(0, 0), (4, 4)]
    goals = [(4, 4), (0, 0)]  # Agents want to swap positions

    # Plan paths
    paths = multi_planner.plan_paths(starts, goals)

    if paths:
        print("Multi-agent paths found:")
        for agent_id, path in paths.items():
            print(f"Agent {agent_id}: {path}")
    else:
        print("No conflict-free paths found")
```

## Practical Exercises

### Exercise 1: Implement a Complete Navigation Stack

**Objective**: Create a complete navigation system that integrates global path planning, local obstacle avoidance, and dynamic obstacle handling.

**Steps**:
1. Implement a global planner (A* or similar)
2. Create a local planner with obstacle avoidance
3. Add dynamic obstacle detection and avoidance
4. Implement path following with feedback control
5. Test the system in simulation with various scenarios

**Expected Outcome**: A functional navigation stack that can plan paths and navigate safely in environments with both static and dynamic obstacles.

### Exercise 2: Compare Path Planning Algorithms

**Objective**: Implement and compare different path planning algorithms in terms of performance and solution quality.

**Steps**:
1. Implement A*, Dijkstra, RRT, and RRT* algorithms
2. Create test environments with various complexity levels
3. Measure computation time, path optimality, and success rate
4. Analyze the trade-offs between different approaches
5. Document findings with visualizations

**Expected Outcome**: A comprehensive comparison of different path planning algorithms with quantitative analysis of their performance characteristics.

### Exercise 3: Multi-Robot Coordination

**Objective**: Implement a multi-robot path planning system that can coordinate multiple agents to avoid conflicts.

**Steps**:
1. Implement a basic multi-agent path planner
2. Add conflict detection and resolution mechanisms
3. Create a simulation environment for testing
4. Test with various numbers of agents and scenarios
5. Evaluate the system's performance and scalability

**Expected Outcome**: A working multi-robot path planning system that can coordinate multiple agents to reach their goals without conflicts.

## Chapter Summary

This chapter covered the fundamental concepts and implementations of path planning and navigation for autonomous robots:

1. **Classical Algorithms**: A* and Dijkstra's algorithms for optimal path planning in known environments, with implementations showing how to handle grid-based representations.

2. **Sampling-Based Methods**: RRT and RRT* algorithms for high-dimensional configuration spaces, demonstrating how random sampling can address complex planning problems.

3. **Navigation Systems**: Integration of global and local planning, showing how to combine long-term path planning with real-time obstacle avoidance.

4. **Obstacle Avoidance**: Local navigation techniques including Vector Field Histogram and Dynamic Window Approach for handling unknown obstacles.

5. **Multi-Agent Planning**: Conflict-Based Search for coordinating multiple robots to avoid conflicts while reaching their goals.

The choice of path planning algorithm depends on the specific requirements of the application, including the environment complexity, real-time constraints, and optimality requirements. Modern robotic systems often combine multiple approaches to achieve robust navigation capabilities.

## Further Reading

1. "Planning Algorithms" by Steven LaValle - Comprehensive coverage of motion planning algorithms
2. "Principles of Robot Motion" by Choset et al. - Theoretical foundations of robot motion planning
3. "Robot Motion Planning" by Latombe - Classic text on configuration space and planning algorithms
4. "Path Planning for Autonomous Vehicles" by Fiorini - Focus on autonomous vehicle navigation
5. "Multi-Robot Systems" by Parker - Coordination and path planning for multiple robots

## Assessment Questions

1. Derive the configuration space for a 2-link planar manipulator and explain its geometric properties.

2. Compare the computational complexity and completeness properties of A*, Dijkstra, and RRT algorithms.

3. Explain the mathematical formulation of the Dynamic Window Approach and its advantages for local navigation.

4. Implement a hybrid A*/DWA navigation system and analyze its performance in various environments.

5. Discuss the challenges in multi-robot path planning and explain how CBS addresses these challenges.

6. Analyze the trade-offs between sampling-based and combinatorial path planning methods.

7. Design a complete navigation pipeline for a mobile robot operating in dynamic environments.

8. Explain how potential field methods work and discuss their limitations for path planning.

9. Describe the implementation of a grid-based path planner with inflation layers for obstacle avoidance.

10. Evaluate the performance of different path planning algorithms in terms of completeness, optimality, and computational efficiency.

