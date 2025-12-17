---
sidebar_position: 14
title: "Chapter 14: Multi-Robot Systems and Coordination"
---

# Chapter 14: Multi-Robot Systems and Coordination

## Learning Objectives

By the end of this chapter, students will be able to:
- Understand the fundamental concepts and architectures of multi-robot systems
- Design coordination strategies for distributed robotic teams
- Implement communication protocols for multi-robot cooperation
- Apply swarm intelligence and collective behavior algorithms
- Evaluate the scalability and robustness of multi-robot systems
- Address challenges in multi-robot task allocation and formation control
- Analyze the trade-offs between centralized and decentralized approaches
- Design fault-tolerant multi-robot systems for real-world deployment

## Theoretical Foundations

### Introduction to Multi-Robot Systems

Multi-robot systems represent a paradigm where multiple autonomous robots collaborate to achieve common goals that would be difficult or impossible for a single robot to accomplish. These systems leverage the principles of distributed computing, collective intelligence, and cooperative control to provide advantages such as:

**Redundancy and Robustness**: Multiple robots can continue operations even when individual robots fail, providing system-level fault tolerance.

**Parallel Processing**: Tasks can be executed in parallel, reducing completion time for large-scale operations.

**Scalability**: Systems can be expanded by adding more robots to handle increased workload.

**Distributed Sensing**: Multiple robots provide broader coverage and more comprehensive environmental information.

**Flexibility**: Teams can adapt their behavior and configuration based on changing requirements.

Multi-robot systems can be classified based on several criteria:

**Homogeneous vs. Heterogeneous**: Homogeneous systems consist of robots with similar capabilities, while heterogeneous systems include robots with different specialized abilities.

**Centralized vs. Decentralized**: Centralized systems rely on a central coordinator, while decentralized systems operate with local decision-making.

**Static vs. Dynamic**: Static systems have fixed team compositions, while dynamic systems can reconfigure based on task requirements.

### Coordination Architectures

Multi-robot coordination can be achieved through various architectural approaches:

**Centralized Coordination**: A central controller collects information from all robots and makes coordination decisions. This approach provides global optimality but suffers from communication bottlenecks and single points of failure.

**Decentralized Coordination**: Each robot makes decisions based on local information and communication with nearby robots. This approach is more robust and scalable but may result in suboptimal global solutions.

**Hierarchical Coordination**: Combines centralized and decentralized approaches with multiple levels of decision-making, balancing optimality and scalability.

**Market-Based Coordination**: Uses economic principles where robots bid for tasks, creating a distributed resource allocation mechanism.

### Communication Models

Effective multi-robot coordination requires reliable communication. Communication models include:

**Broadcast Communication**: All robots receive the same message, useful for global updates.

**Point-to-Point Communication**: Direct communication between specific robots, reducing network congestion.

**Multi-hop Communication**: Messages are relayed through intermediate robots, extending communication range.

**Opportunistic Communication**: Communication occurs when robots are in proximity, suitable for dynamic environments.

## Communication Protocols and Architectures

### Robot Communication Networks

```python
#!/usr/bin/env python3

import socket
import threading
import json
import time
import random
from typing import Dict, List, Tuple, Optional
import queue
import hashlib

class RobotCommunicationNode:
    def __init__(self, robot_id: str, host: str = 'localhost', port: int = 0):
        self.robot_id = robot_id
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # If port is 0, let the OS assign a free port
        if port == 0:
            self.socket.bind((host, 0))
            self.port = self.socket.getsockname()[1]
        else:
            self.socket.bind((host, port))

        self.neighbors = {}  # {robot_id: (host, port)}
        self.message_queue = queue.Queue()
        self.running = True
        self.message_handlers = {}

        # Start listening thread
        self.listen_thread = threading.Thread(target=self._listen_loop)
        self.listen_thread.daemon = True
        self.listen_thread.start()

        print(f"Robot {self.robot_id} listening on {self.host}:{self.port}")

    def register_message_handler(self, message_type: str, handler):
        """Register a handler for specific message types"""
        self.message_handlers[message_type] = handler

    def send_message(self, robot_id: str, message: Dict):
        """Send message to a specific robot"""
        if robot_id in self.neighbors:
            host, port = self.neighbors[robot_id]
            message['from_robot'] = self.robot_id
            message['timestamp'] = time.time()
            message_json = json.dumps(message)
            self.socket.sendto(message_json.encode(), (host, port))

    def broadcast_message(self, message: Dict):
        """Broadcast message to all known neighbors"""
        message['from_robot'] = self.robot_id
        message['timestamp'] = time.time()
        message['broadcast'] = True
        message_json = json.dumps(message)

        for robot_id, (host, port) in self.neighbors.items():
            try:
                self.socket.sendto(message_json.encode(), (host, port))
            except Exception as e:
                print(f"Error broadcasting to {robot_id}: {e}")

    def add_neighbor(self, robot_id: str, host: str, port: int):
        """Add a neighboring robot to communication network"""
        self.neighbors[robot_id] = (host, port)

    def _listen_loop(self):
        """Main listening loop for incoming messages"""
        while self.running:
            try:
                data, addr = self.socket.recvfrom(4096)
                message = json.loads(data.decode())

                # Add source information
                message['source_addr'] = addr

                # Handle the message
                self._handle_message(message)

            except socket.timeout:
                continue
            except Exception as e:
                print(f"Error receiving message: {e}")

    def _handle_message(self, message: Dict):
        """Handle incoming message based on type"""
        msg_type = message.get('type', 'unknown')

        if msg_type in self.message_handlers:
            self.message_handlers[msg_type](message)
        else:
            print(f"Unknown message type: {msg_type} from {message.get('from_robot')}")

    def close(self):
        """Close the communication node"""
        self.running = False
        self.socket.close()

class MultiRobotNetwork:
    def __init__(self):
        self.robots = {}  # {robot_id: RobotCommunicationNode}
        self.topology = {}  # {robot_id: [neighbor_ids]}
        self.network_lock = threading.Lock()

    def add_robot(self, robot_id: str, host: str = 'localhost', port: int = 0) -> RobotCommunicationNode:
        """Add a robot to the network"""
        with self.network_lock:
            if robot_id in self.robots:
                raise ValueError(f"Robot {robot_id} already exists in network")

            node = RobotCommunicationNode(robot_id, host, port)
            self.robots[robot_id] = node
            self.topology[robot_id] = []

            # Connect to existing robots (simple star topology)
            for existing_robot_id, existing_node in self.robots.items():
                if existing_robot_id != robot_id:
                    # Add bidirectional connection
                    node.add_neighbor(existing_robot_id, existing_node.host, existing_node.port)
                    existing_node.add_neighbor(robot_id, node.host, node.port)
                    self.topology[robot_id].append(existing_robot_id)
                    self.topology[existing_robot_id].append(robot_id)

            return node

    def get_robot(self, robot_id: str) -> RobotCommunicationNode:
        """Get robot communication node"""
        return self.robots.get(robot_id)

    def remove_robot(self, robot_id: str):
        """Remove robot from network"""
        with self.network_lock:
            if robot_id in self.robots:
                # Remove from neighbors
                for neighbor_id in self.topology[robot_id]:
                    if neighbor_id in self.robots:
                        # Remove this robot from neighbor's neighbor list
                        if robot_id in self.robots[neighbor_id].neighbors:
                            del self.robots[neighbor_id].neighbors[robot_id]

                # Remove robot
                node = self.robots[robot_id]
                node.close()
                del self.robots[robot_id]
                del self.topology[robot_id]

    def broadcast_to_all(self, message: Dict):
        """Broadcast message to all robots in network"""
        for robot_node in self.robots.values():
            robot_node.broadcast_message(message)

    def get_network_status(self) -> Dict:
        """Get current network status"""
        status = {
            'robot_count': len(self.robots),
            'robots': list(self.robots.keys()),
            'topology': self.topology.copy()
        }
        return status

# Example message handlers
def position_update_handler(message: Dict):
    """Handle position update messages"""
    robot_id = message.get('from_robot')
    position = message.get('position', [0, 0, 0])
    timestamp = message.get('timestamp')
    print(f"Position update from {robot_id}: {position} at {timestamp}")

def task_assignment_handler(message: Dict):
    """Handle task assignment messages"""
    robot_id = message.get('from_robot')
    task = message.get('task')
    print(f"Task assignment from {robot_id}: {task}")

# Example usage
if __name__ == "__main__":
    # Create multi-robot network
    network = MultiRobotNetwork()

    # Add robots to network
    robot1 = network.add_robot('R1')
    robot2 = network.add_robot('R2')
    robot3 = network.add_robot('R3')

    # Register message handlers
    robot1.register_message_handler('position_update', position_update_handler)
    robot1.register_message_handler('task_assignment', task_assignment_handler)
    robot2.register_message_handler('position_update', position_update_handler)
    robot3.register_message_handler('position_update', position_update_handler)

    # Simulate position updates
    position_msg = {
        'type': 'position_update',
        'position': [1.0, 2.0, 0.5],
        'orientation': [0, 0, 0, 1]
    }

    robot1.broadcast_message(position_msg)

    # Simulate task assignment
    task_msg = {
        'type': 'task_assignment',
        'task_id': 'T001',
        'task_description': 'Patrol area A',
        'priority': 1
    }

    robot1.send_message('R2', task_msg)

    print("Network status:", network.get_network_status())

    # Cleanup
    network.remove_robot('R1')
    network.remove_robot('R2')
    network.remove_robot('R3')
```

### Consensus Algorithms

Consensus algorithms enable multi-robot systems to agree on values or decisions:

```python
#!/usr/bin/env python3

import numpy as np
import threading
import time
from typing import Dict, List, Callable
import random

class RobotConsensusNode:
    def __init__(self, robot_id: str, initial_value: float, neighbors: List[str]):
        self.robot_id = robot_id
        self.current_value = initial_value
        self.neighbors = neighbors
        self.values_from_neighbors = {}
        self.consensus_reached = False
        self.iteration_count = 0
        self.max_iterations = 100
        self.convergence_threshold = 0.001
        self.lock = threading.Lock()

        # For averaging consensus
        self.weight_matrix = self._initialize_weights()

    def _initialize_weights(self) -> Dict[str, float]:
        """Initialize weights for consensus algorithm"""
        weights = {}
        n_neighbors = len(self.neighbors) + 1  # Include self

        # Equal weights (Metropolis-Hastings weights)
        for neighbor in self.neighbors:
            weights[neighbor] = 1.0 / n_neighbors

        # Self-weight
        weights[self.robot_id] = 1.0 / n_neighbors

        return weights

    def update_consensus(self, neighbor_values: Dict[str, float]) -> float:
        """Update consensus value using weighted averaging"""
        with self.lock:
            # Update values from neighbors
            for neighbor, value in neighbor_values.items():
                if neighbor in self.neighbors:
                    self.values_from_neighbors[neighbor] = value

            # Perform weighted averaging
            new_value = self.weight_matrix[self.robot_id] * self.current_value

            for neighbor in self.neighbors:
                if neighbor in self.values_from_neighbors:
                    new_value += self.weight_matrix[neighbor] * self.values_from_neighbors[neighbor]

            self.current_value = new_value
            self.iteration_count += 1

            return self.current_value

    def get_current_value(self) -> float:
        """Get current consensus value"""
        with self.lock:
            return self.current_value

    def is_converged(self, global_values: List[float]) -> bool:
        """Check if consensus is reached"""
        if len(global_values) < 2:
            return True

        max_val = max(global_values)
        min_val = min(global_values)

        with self.lock:
            self.consensus_reached = (max_val - min_val) < self.convergence_threshold
            return self.consensus_reached

class DistributedAveraging:
    def __init__(self, robot_ids: List[str], initial_values: List[float]):
        """Initialize distributed averaging system"""
        self.robot_ids = robot_ids
        self.nodes = {}

        # Create communication topology (simple ring topology)
        for i, robot_id in enumerate(robot_ids):
            # Each robot communicates with neighbors in ring
            left_neighbor = robot_ids[(i - 1) % len(robot_ids)]
            right_neighbor = robot_ids[(i + 1) % len(robot_ids)]
            neighbors = [left_neighbor, right_neighbor]

            node = RobotConsensusNode(
                robot_id,
                initial_values[i],
                neighbors
            )
            self.nodes[robot_id] = node

    def run_consensus(self, max_iterations: int = 50) -> List[float]:
        """Run consensus algorithm"""
        for iteration in range(max_iterations):
            # Collect current values from all nodes
            current_values = {}
            for robot_id, node in self.nodes.items():
                current_values[robot_id] = node.get_current_value()

            # Update each node with values from neighbors
            for robot_id, node in self.nodes.items():
                # Get values from this node's neighbors
                neighbor_values = {
                    neighbor: current_values[neighbor]
                    for neighbor in node.neighbors
                    if neighbor in current_values
                }

                # Update the node
                node.update_consensus(neighbor_values)

            # Check for convergence
            all_values = [node.get_current_value() for node in self.nodes.values()]
            max_diff = max(all_values) - min(all_values)

            print(f"Iteration {iteration + 1}: Max difference = {max_diff:.6f}")

            if max_diff < 0.001:  # Convergence threshold
                print(f"Consensus reached after {iteration + 1} iterations")
                break

        final_values = [node.get_current_value() for node in self.nodes.values()]
        return final_values

class TaskAllocationConsensus:
    def __init__(self, robot_ids: List[str], task_ids: List[str]):
        """Initialize task allocation using consensus"""
        self.robot_ids = robot_ids
        self.task_ids = task_ids
        self.task_assignments = {task_id: None for task_id in task_ids}  # {task_id: robot_id}
        self.robot_loads = {robot_id: 0 for robot_id in robot_ids}
        self.task_preferences = {}  # {robot_id: {task_id: preference_score}}

        # Initialize random preferences
        for robot_id in robot_ids:
            self.task_preferences[robot_id] = {}
            for task_id in task_ids:
                self.task_preferences[robot_id][task_id] = random.random()

    def allocate_tasks(self) -> Dict[str, str]:
        """Allocate tasks to robots using consensus-based approach"""
        assignments = {}

        # Simple greedy allocation with load balancing
        for task_id in self.task_ids:
            best_robot = None
            best_score = -1

            for robot_id in self.robot_ids:
                # Calculate score based on preference and current load
                preference = self.task_preferences[robot_id][task_id]
                load_factor = 1.0 / (1.0 + self.robot_loads[robot_id])  # Prefer less loaded robots
                score = preference * load_factor

                if score > best_score:
                    best_score = score
                    best_robot = robot_id

            if best_robot:
                assignments[task_id] = best_robot
                self.robot_loads[best_robot] += 1
                self.task_assignments[task_id] = best_robot

        return assignments

class FormationControl:
    def __init__(self, robot_ids: List[str], formation_shape: str = "line"):
        """Initialize formation control system"""
        self.robot_ids = robot_ids
        self.formation_shape = formation_shape
        self.robot_positions = {robot_id: np.array([0.0, 0.0]) for robot_id in robot_ids}
        self.desired_positions = self._calculate_desired_positions()
        self.velocity_commands = {robot_id: np.array([0.0, 0.0]) for robot_id in robot_ids}

    def _calculate_desired_positions(self) -> Dict[str, np.ndarray]:
        """Calculate desired positions for the formation"""
        n_robots = len(self.robot_ids)
        desired_positions = {}

        if self.formation_shape == "line":
            spacing = 1.0
            for i, robot_id in enumerate(self.robot_ids):
                desired_positions[robot_id] = np.array([i * spacing, 0.0])
        elif self.formation_shape == "circle":
            radius = 2.0
            for i, robot_id in enumerate(self.robot_ids):
                angle = 2 * np.pi * i / n_robots
                desired_positions[robot_id] = np.array([
                    radius * np.cos(angle),
                    radius * np.sin(angle)
                ])
        elif self.formation_shape == "triangle":
            for i, robot_id in enumerate(self.robot_ids):
                if i == 0:
                    pos = np.array([0.0, 1.0])
                elif i == 1:
                    pos = np.array([-0.866, -0.5])  # cos(240°), sin(240°)
                elif i == 2:
                    pos = np.array([0.866, -0.5])   # cos(300°), sin(300°)
                else:
                    # For more than 3 robots in triangle, arrange in multiple triangles
                    angle = 2 * np.pi * (i - 3) / (n_robots - 3) if n_robots > 3 else 0
                    pos = np.array([np.cos(angle), np.sin(angle)]) * (i // 3 + 1)

                desired_positions[robot_id] = pos

        return desired_positions

    def update_formation(self, dt: float = 0.1):
        """Update robot positions to maintain formation"""
        for robot_id in self.robot_ids:
            current_pos = self.robot_positions[robot_id]
            desired_pos = self.desired_positions[robot_id]

            # Calculate control vector (simple proportional control)
            error = desired_pos - current_pos
            control_vector = 2.0 * error  # Proportional gain

            # Update velocity
            self.velocity_commands[robot_id] = control_vector

            # Update position (simple integration)
            self.robot_positions[robot_id] += control_vector * dt

    def get_robot_positions(self) -> Dict[str, np.ndarray]:
        """Get current robot positions"""
        return self.robot_positions.copy()

    def get_velocity_commands(self) -> Dict[str, np.ndarray]:
        """Get velocity commands for robots"""
        return self.velocity_commands.copy()

# Example usage
if __name__ == "__main__":
    print("Testing Multi-Robot Consensus and Coordination...")

    # Test distributed averaging
    robot_ids = ['R1', 'R2', 'R3', 'R4']
    initial_values = [10.0, 20.0, 5.0, 15.0]

    print("Initial values:", dict(zip(robot_ids, initial_values)))

    averaging_system = DistributedAveraging(robot_ids, initial_values)
    final_values = averaging_system.run_consensus()

    print("Final consensus values:", dict(zip(robot_ids, final_values)))
    print("Expected average:", sum(initial_values) / len(initial_values))

    # Test task allocation
    task_ids = ['T1', 'T2', 'T3', 'T4']
    task_allocator = TaskAllocationConsensus(robot_ids, task_ids)
    assignments = task_allocator.allocate_tasks()

    print("Task assignments:", assignments)

    # Test formation control
    formation_controller = FormationControl(robot_ids[:3], formation_shape="triangle")
    print("Initial positions:", formation_controller.get_robot_positions())

    # Update formation
    for step in range(10):
        formation_controller.update_formation(dt=0.1)

    print("Final positions:", formation_controller.get_robot_positions())
```

## Coordination Strategies

### Market-Based Coordination

Market-based coordination uses economic principles to allocate tasks among robots:

```python
#!/usr/bin/env python3

import heapq
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random

@dataclass
class Task:
    id: str
    location: Tuple[float, float]
    value: float  # Benefit of completing task
    deadline: float  # Time by which task should be completed
    required_capabilities: List[str]
    estimated_completion_time: float

@dataclass
class RobotCapability:
    id: str
    position: Tuple[float, float]
    capabilities: List[str]
    battery_level: float
    max_speed: float

class MarketBid:
    def __init__(self, robot_id: str, task_id: str, bid_amount: float,
                 expected_completion_time: float):
        self.robot_id = robot_id
        self.task_id = task_id
        self.bid_amount = bid_amount
        self.expected_completion_time = expected_completion_time
        self.timestamp = time.time()

    def __lt__(self, other):
        # For priority queue: lower bid amount is better
        return self.bid_amount < other.bid_amount

class MarketBasedCoordinator:
    def __init__(self, robots: List[RobotCapability]):
        self.robots = {robot.id: robot for robot in robots}
        self.tasks = {}
        self.assigned_tasks = {}  # {task_id: robot_id}
        self.robot_assignments = {}  # {robot_id: [task_ids]}
        self.completed_tasks = []
        self.task_queue = []
        self.time = 0.0

    def add_task(self, task: Task):
        """Add a task to the market"""
        self.tasks[task.id] = task
        heapq.heappush(self.task_queue, (task.deadline, task.id))

    def calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points"""
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5

    def calculate_robot_cost(self, robot: RobotCapability, task: Task) -> float:
        """Calculate the cost for a robot to complete a task"""
        # Distance cost
        distance = self.calculate_distance(robot.position, task.location)
        time_to_reach = distance / robot.max_speed

        # Battery cost (simplified)
        battery_cost = (1 - robot.battery_level) * 10

        # Capability match bonus
        capability_bonus = 0
        for cap in task.required_capabilities:
            if cap not in robot.capabilities:
                return float('inf')  # Robot cannot perform task
            else:
                capability_bonus -= 2  # Small bonus for capability match

        # Total cost (negative because we want to minimize cost)
        total_cost = distance + time_to_reach + battery_cost + capability_bonus
        return total_cost

    def generate_bids(self) -> List[MarketBid]:
        """Generate bids from all robots for all available tasks"""
        bids = []

        for robot_id, robot in self.robots.items():
            for task_id, task in self.tasks.items():
                if task_id in self.assigned_tasks:
                    continue  # Task already assigned

                # Check if robot can perform task
                can_perform = True
                for cap in task.required_capabilities:
                    if cap not in robot.capabilities:
                        can_perform = False
                        break

                if can_perform:
                    # Calculate cost and generate bid
                    cost = self.calculate_robot_cost(robot, task)
                    estimated_time = cost / robot.max_speed  # Simplified

                    # Bid amount: value - cost (robot bids how much it wants to earn)
                    bid_amount = task.value - cost

                    if bid_amount > 0:  # Only bid if profitable
                        bid = MarketBid(robot_id, task_id, bid_amount, estimated_time)
                        bids.append(bid)

        return bids

    def allocate_tasks(self) -> Dict[str, str]:
        """Allocate tasks using market-based bidding"""
        bids = self.generate_bids()

        # Sort bids by bid amount (descending - highest bidders first)
        bids.sort(key=lambda x: x.bid_amount, reverse=True)

        assignments = {}
        assigned_robots = set()
        assigned_tasks = set()

        for bid in bids:
            if (bid.task_id not in assigned_tasks and
                bid.robot_id not in assigned_robots and
                bid.task_id in self.tasks):

                assignments[bid.task_id] = bid.robot_id
                assigned_tasks.add(bid.task_id)
                assigned_robots.add(bid.robot_id)

        # Update system with assignments
        for task_id, robot_id in assignments.items():
            self.assigned_tasks[task_id] = robot_id

            if robot_id not in self.robot_assignments:
                self.robot_assignments[robot_id] = []
            self.robot_assignments[robot_id].append(task_id)

        return assignments

    def complete_task(self, task_id: str):
        """Mark a task as completed"""
        if task_id in self.tasks:
            self.completed_tasks.append(task_id)
            if task_id in self.assigned_tasks:
                robot_id = self.assigned_tasks[task_id]
                if robot_id in self.robot_assignments:
                    self.robot_assignments[robot_id].remove(task_id)
                del self.assigned_tasks[task_id]
            del self.tasks[task_id]

class AuctionBasedCoordinator:
    def __init__(self, robots: List[RobotCapability]):
        self.robots = {robot.id: robot for robot in robots}
        self.tasks = {}
        self.current_auction_task = None
        self.bids = {}
        self.assignments = {}
        self.auction_results = []

    def start_auction(self, task: Task):
        """Start an auction for a specific task"""
        self.current_auction_task = task
        self.bids = {}

    def submit_bid(self, robot_id: str, bid_amount: float):
        """Submit a bid for the current auction"""
        if self.current_auction_task and robot_id in self.robots:
            self.bids[robot_id] = bid_amount

    def close_auction(self) -> Optional[Tuple[str, float]]:
        """Close the auction and return winner"""
        if not self.bids:
            return None

        # Find the highest bidder
        winner = max(self.bids.keys(), key=lambda x: self.bids[x])
        winning_bid = self.bids[winner]

        # Record auction result
        result = {
            'task_id': self.current_auction_task.id,
            'winner': winner,
            'winning_bid': winning_bid,
            'timestamp': time.time()
        }
        self.auction_results.append(result)

        # Update assignments
        self.assignments[self.current_auction_task.id] = winner

        # Clear current auction
        self.current_auction_task = None
        self.bids = {}

        return winner, winning_bid

class CoalitionFormation:
    def __init__(self, robots: List[RobotCapability]):
        self.robots = robots
        self.coalitions = []
        self.robot_coalition_map = {}

    def calculate_coalition_value(self, robot_ids: List[str], task: Task) -> float:
        """Calculate the value of a coalition for a task"""
        # Simplified value calculation
        # In practice, this would consider combined capabilities, efficiency, etc.
        total_capability_score = 0

        for robot_id in robot_ids:
            robot = next((r for r in self.robots if r.id == robot_id), None)
            if robot:
                # Score based on relevant capabilities
                for cap in task.required_capabilities:
                    if cap in robot.capabilities:
                        total_capability_score += 1

        # Consider distance and coordination overhead
        avg_distance = 0
        if len(robot_ids) > 1:
            positions = [next((r.position for r in self.robots if r.id == rid), (0,0))
                        for rid in robot_ids]
            for i in range(len(positions)):
                for j in range(i+1, len(positions)):
                    avg_distance += self._distance(positions[i], positions[j])
            avg_distance /= (len(robot_ids) * (len(robot_ids) - 1) / 2)

        overhead_penalty = avg_distance * 0.1  # Penalty for high coordination overhead

        return total_capability_score - overhead_penalty

    def _distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate distance between two points"""
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5

    def form_coalitions(self, tasks: List[Task]) -> List[List[str]]:
        """Form coalitions of robots for tasks"""
        coalitions = []

        for task in tasks:
            # Find all possible coalitions for this task
            best_coalition = []
            best_value = float('-inf')

            # Try different coalition sizes (simplified: just pairs for now)
            for i, robot1 in enumerate(self.robots):
                for j, robot2 in enumerate(self.robots[i+1:], i+1):
                    coalition = [robot1.id, robot2.id]
                    value = self.calculate_coalition_value(coalition, task)

                    if value > best_value:
                        best_value = value
                        best_coalition = coalition

            if best_coalition and best_value > 0:
                coalitions.append(best_coalition)

                # Update robot coalition mapping
                for robot_id in best_coalition:
                    self.robot_coalition_map[robot_id] = best_coalition

        return coalitions

# Example usage
if __name__ == "__main__":
    print("Testing Market-Based Coordination...")

    # Create robots
    robots = [
        RobotCapability("R1", (0, 0), ["navigation", "manipulation"], 0.9, 1.0),
        RobotCapability("R2", (5, 0), ["navigation", "sensing"], 0.8, 1.2),
        RobotCapability("R3", (0, 5), ["navigation", "manipulation", "sensing"], 0.7, 0.8),
        RobotCapability("R4", (5, 5), ["navigation"], 0.95, 1.5)
    ]

    # Create tasks
    tasks = [
        Task("T1", (1, 1), 10.0, 100.0, ["navigation", "manipulation"], 5.0),
        Task("T2", (4, 1), 8.0, 80.0, ["navigation", "sensing"], 3.0),
        Task("T3", (1, 4), 12.0, 120.0, ["navigation", "manipulation", "sensing"], 7.0),
        Task("T4", (4, 4), 6.0, 60.0, ["navigation"], 2.0)
    ]

    # Test market-based coordination
    market_coord = MarketBasedCoordinator(robots)

    for task in tasks:
        market_coord.add_task(task)

    assignments = market_coord.allocate_tasks()
    print("Market-based task assignments:", assignments)

    # Test auction-based coordination
    auction_coord = AuctionBasedCoordinator(robots)

    for task in tasks[:2]:  # Just first 2 tasks for auction example
        auction_coord.start_auction(task)

        # Robots submit bids (simulated)
        for robot in robots:
            capability_match = len(set(task.required_capabilities) & set(robot.capabilities))
            distance = ((robot.position[0] - task.location[0])**2 +
                       (robot.position[1] - task.location[1])**2)**0.5
            bid = task.value - distance + capability_match * 2
            auction_coord.submit_bid(robot.id, bid)

        winner = auction_coord.close_auction()
        if winner:
            print(f"Auction winner for {task.id}: {winner[0]} with bid {winner[1]:.2f}")

    # Test coalition formation
    coalition_former = CoalitionFormation(robots)
    coalitions = coalition_former.form_coalitions(tasks[:2])  # First 2 tasks
    print("Formed coalitions:", coalitions)
```

### Swarm Intelligence Algorithms

Swarm intelligence algorithms are inspired by the collective behavior of social insects:

```python
#!/usr/bin/env python3

import numpy as np
import random
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class Particle:
    """Particle for Particle Swarm Optimization"""
    position: np.ndarray
    velocity: np.ndarray
    best_position: np.ndarray
    best_fitness: float
    current_fitness: float

    def __init__(self, dimensions: int, bounds: Tuple[float, float]):
        self.position = np.random.uniform(bounds[0], bounds[1], dimensions)
        self.velocity = np.random.uniform(-1, 1, dimensions)
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')
        self.current_fitness = float('inf')

class ParticleSwarmOptimizer:
    def __init__(self, dimensions: int, bounds: Tuple[float, float],
                 num_particles: int = 30, w: float = 0.7, c1: float = 1.5, c2: float = 1.5):
        self.dimensions = dimensions
        self.bounds = bounds
        self.num_particles = num_particles
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive parameter
        self.c2 = c2  # Social parameter

        # Initialize particles
        self.particles = [
            Particle(dimensions, bounds) for _ in range(num_particles)
        ]

        # Global best
        self.global_best_position = None
        self.global_best_fitness = float('inf')

    def objective_function(self, x: np.ndarray) -> float:
        """Example objective function (can be replaced with specific problem)"""
        # Sphere function - minimum at origin
        return np.sum(x**2)

    def update_particle(self, particle: Particle):
        """Update a single particle"""
        # Calculate fitness
        fitness = self.objective_function(particle.position)
        particle.current_fitness = fitness

        # Update personal best
        if fitness < particle.best_fitness:
            particle.best_fitness = fitness
            particle.best_position = particle.position.copy()

        # Update global best
        if fitness < self.global_best_fitness:
            self.global_best_fitness = fitness
            self.global_best_position = particle.position.copy()

        # Update velocity
        r1, r2 = random.random(), random.random()

        cognitive_component = self.c1 * r1 * (particle.best_position - particle.position)
        social_component = self.c2 * r2 * (self.global_best_position - particle.position)

        particle.velocity = (self.w * particle.velocity +
                           cognitive_component + social_component)

        # Update position
        particle.position += particle.velocity

        # Apply bounds
        particle.position = np.clip(particle.position, self.bounds[0], self.bounds[1])

    def optimize(self, max_iterations: int = 100) -> Tuple[np.ndarray, float]:
        """Run PSO optimization"""
        fitness_history = []

        for iteration in range(max_iterations):
            for particle in self.particles:
                self.update_particle(particle)

            fitness_history.append(self.global_best_fitness)

            if iteration % 20 == 0:
                print(f"Iteration {iteration}: Best fitness = {self.global_best_fitness:.6f}")

        return self.global_best_position, self.global_best_fitness

class AntColonyOptimizer:
    def __init__(self, num_ants: int, num_cities: int, alpha: float = 1.0,
                 beta: float = 2.0, evaporation_rate: float = 0.5):
        self.num_ants = num_ants
        self.num_cities = num_cities
        self.alpha = alpha  # Pheromone importance
        self.beta = beta    # Heuristic importance
        self.evaporation_rate = evaporation_rate

        # Initialize pheromone matrix
        self.pheromones = np.ones((num_cities, num_cities)) * 0.1

        # Initialize distance matrix (for TSP example)
        self.distances = self._generate_distance_matrix()

        # Best solution found
        self.best_path = None
        self.best_distance = float('inf')

    def _generate_distance_matrix(self) -> np.ndarray:
        """Generate random distance matrix for TSP"""
        # Generate random city coordinates
        self.city_coordinates = np.random.rand(self.num_cities, 2)

        # Calculate distance matrix
        distances = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != j:
                    dx = self.city_coordinates[i][0] - self.city_coordinates[j][0]
                    dy = self.city_coordinates[i][1] - self.city_coordinates[j][1]
                    distances[i][j] = np.sqrt(dx**2 + dy**2)

        return distances

    def calculate_path_distance(self, path: List[int]) -> float:
        """Calculate total distance for a path"""
        total_distance = 0
        for i in range(len(path)):
            from_city = path[i]
            to_city = path[(i + 1) % len(path)]  # Return to start for TSP
            total_distance += self.distances[from_city][to_city]
        return total_distance

    def select_next_city(self, current_city: int, visited: List[int]) -> int:
        """Select next city based on pheromone and heuristic information"""
        unvisited = [i for i in range(self.num_cities) if i not in visited]

        if not unvisited:
            return visited[0]  # Return to start

        # Calculate probabilities
        probabilities = []
        total = 0

        for city in unvisited:
            pheromone = self.pheromones[current_city][city] ** self.alpha
            heuristic = (1 / self.distances[current_city][city]) ** self.beta
            probability = pheromone * heuristic
            probabilities.append(probability)
            total += probability

        if total == 0:
            return random.choice(unvisited)

        # Normalize probabilities
        probabilities = [p / total for p in probabilities]

        # Select based on probabilities
        r = random.random()
        cumulative = 0
        for i, prob in enumerate(probabilities):
            cumulative += prob
            if r <= cumulative:
                return unvisited[i]

        return unvisited[-1]

    def run_ant_tour(self) -> List[int]:
        """Run a single ant tour"""
        start_city = random.randint(0, self.num_cities - 1)
        tour = [start_city]
        visited = {start_city}

        current_city = start_city
        while len(tour) < self.num_cities:
            next_city = self.select_next_city(current_city, list(visited))
            tour.append(next_city)
            visited.add(next_city)
            current_city = next_city

        return tour

    def update_pheromones(self, ant_paths: List[List[int]]):
        """Update pheromone levels"""
        # Evaporate pheromones
        self.pheromones *= (1 - self.evaporation_rate)

        # Add pheromones based on tour quality
        for path in ant_paths:
            distance = self.calculate_path_distance(path)
            if distance < self.best_distance:
                self.best_distance = distance
                self.best_path = path.copy()

            pheromone_deposit = 1 / distance

            for i in range(len(path)):
                from_city = path[i]
                to_city = path[(i + 1) % len(path)]
                self.pheromones[from_city][to_city] += pheromone_deposit

    def optimize(self, max_iterations: int = 100) -> Tuple[List[int], float]:
        """Run ACO optimization"""
        for iteration in range(max_iterations):
            ant_paths = []

            # Each ant constructs a solution
            for ant in range(self.num_ants):
                path = self.run_ant_tour()
                ant_paths.append(path)

            # Update pheromones based on all ant paths
            self.update_pheromones(ant_paths)

            if iteration % 20 == 0:
                print(f"Iteration {iteration}: Best distance = {self.best_distance:.4f}")

        return self.best_path, self.best_distance

class BeeColonyOptimizer:
    """Artificial Bee Colony (ABC) Algorithm"""
    def __init__(self, dimensions: int, bounds: Tuple[float, float],
                 num_employed: int = 20, num_onlookers: int = 20,
                 max_trials: int = 100):
        self.dimensions = dimensions
        self.bounds = bounds
        self.num_employed = num_employed
        self.num_onlookers = num_onlookers
        self.max_trials = max_trials

        # Initialize food sources (solutions)
        self.food_sources = []
        self.fitness_values = []
        self.trial_counts = []

        for _ in range(num_employed):
            solution = np.random.uniform(bounds[0], bounds[1], dimensions)
            self.food_sources.append(solution)
            self.fitness_values.append(self._calculate_fitness(solution))
            self.trial_counts.append(0)

        self.best_solution = None
        self.best_fitness = float('-inf')
        self._update_best()

    def _calculate_fitness(self, solution: np.ndarray) -> float:
        """Calculate fitness (for maximization problems)"""
        # Convert objective function (minimization) to fitness (maximization)
        obj_val = np.sum(solution**2)  # Example objective function
        if obj_val == 0:
            return float('inf')
        return 1 / (1 + obj_val)  # Fitness function

    def _update_best(self):
        """Update best solution"""
        best_idx = np.argmax(self.fitness_values)
        if self.fitness_values[best_idx] > self.best_fitness:
            self.best_fitness = self.fitness_values[best_idx]
            self.best_solution = self.food_sources[best_idx].copy()

    def optimize(self, max_iterations: int = 100) -> Tuple[np.ndarray, float]:
        """Run ABC optimization"""
        for iteration in range(max_iterations):
            # Employed bee phase
            for i in range(self.num_employed):
                # Generate new solution
                k = random.randint(0, self.num_employed - 1)
                while k == i:
                    k = random.randint(0, self.num_employed - 1)

                phi = random.uniform(-1, 1)
                new_solution = (self.food_sources[i] +
                               phi * (self.food_sources[i] - self.food_sources[k]))

                # Apply bounds
                new_solution = np.clip(new_solution, self.bounds[0], self.bounds[1])

                # Evaluate new solution
                new_fitness = self._calculate_fitness(new_solution)

                # Greedy selection
                if new_fitness > self.fitness_values[i]:
                    self.food_sources[i] = new_solution
                    self.fitness_values[i] = new_fitness
                    self.trial_counts[i] = 0
                else:
                    self.trial_counts[i] += 1

            # Calculate selection probabilities for onlooker bees
            max_fitness = max(self.fitness_values)
            if max_fitness > 0:
                probabilities = [f / max_fitness for f in self.fitness_values]
            else:
                probabilities = [1.0 / self.num_employed] * self.num_employed

            # Onlooker bee phase
            for _ in range(self.num_onlookers):
                # Select food source based on probability
                r = random.random()
                selected = 0
                cumulative = probabilities[0]

                while r > cumulative and selected < self.num_employed - 1:
                    selected += 1
                    cumulative += probabilities[selected]

                # Generate new solution for selected food source
                k = random.randint(0, self.num_employed - 1)
                while k == selected:
                    k = random.randint(0, self.num_employed - 1)

                phi = random.uniform(-1, 1)
                new_solution = (self.food_sources[selected] +
                               phi * (self.food_sources[selected] - self.food_sources[k]))

                # Apply bounds
                new_solution = np.clip(new_solution, self.bounds[0], self.bounds[1])

                # Evaluate new solution
                new_fitness = self._calculate_fitness(new_solution)

                # Greedy selection
                if new_fitness > self.fitness_values[selected]:
                    self.food_sources[selected] = new_solution
                    self.fitness_values[selected] = new_fitness
                    self.trial_counts[selected] = 0
                else:
                    self.trial_counts[selected] += 1

            # Scout bee phase (replace abandoned solutions)
            for i in range(self.num_employed):
                if self.trial_counts[i] >= self.max_trials:
                    # Generate new random solution
                    self.food_sources[i] = np.random.uniform(
                        self.bounds[0], self.bounds[1], self.dimensions
                    )
                    self.fitness_values[i] = self._calculate_fitness(self.food_sources[i])
                    self.trial_counts[i] = 0

            # Update best solution
            self._update_best()

            if iteration % 20 == 0:
                print(f"Iteration {iteration}: Best fitness = {self.best_fitness:.6f}")

        return self.best_solution, self.best_fitness

# Example usage
if __name__ == "__main__":
    print("Testing Swarm Intelligence Algorithms...")

    # Test Particle Swarm Optimization
    print("\nParticle Swarm Optimization:")
    pso = ParticleSwarmOptimizer(dimensions=2, bounds=(-5, 5), num_particles=30)
    best_pos, best_fit = pso.optimize(max_iterations=100)
    print(f"Best position: {best_pos}, Best fitness: {best_fit}")

    # Test Ant Colony Optimization
    print("\nAnt Colony Optimization (TSP):")
    aco = AntColonyOptimizer(num_ants=20, num_cities=10)
    best_path, best_dist = aco.optimize(max_iterations=100)
    print(f"Best path: {best_path}, Best distance: {best_dist:.4f}")

    # Test Artificial Bee Colony
    print("\nArtificial Bee Colony:")
    abc = BeeColonyOptimizer(dimensions=2, bounds=(-5, 5))
    best_sol, best_fit_val = abc.optimize(max_iterations=100)
    print(f"Best solution: {best_sol}, Best fitness: {best_fit_val:.6f}")
```

## Formation Control and Task Allocation

### Formation Control Algorithms

```python
#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import math

class FormationController:
    def __init__(self, robot_ids: List[str]):
        self.robot_ids = robot_ids
        self.robot_positions = {robot_id: np.array([0.0, 0.0]) for robot_id in robot_ids}
        self.robot_velocities = {robot_id: np.array([0.0, 0.0]) for robot_id in robot_ids}
        self.desired_formation = {}
        self.neighbors = {robot_id: [] for robot_id in robot_ids}
        self.formation_active = False

    def set_formation(self, formation_type: str, leader_position: np.ndarray = None):
        """Set the desired formation pattern"""
        n_robots = len(self.robot_ids)

        if formation_type == "line":
            self._create_line_formation()
        elif formation_type == "diamond":
            self._create_diamond_formation()
        elif formation_type == "circle":
            self._create_circle_formation()
        elif formation_type == "v_formation":
            self._create_v_formation()
        elif formation_type == "grid":
            self._create_grid_formation()

        self.formation_active = True

    def _create_line_formation(self):
        """Create a line formation"""
        spacing = 1.0
        for i, robot_id in enumerate(self.robot_ids):
            self.desired_formation[robot_id] = np.array([i * spacing, 0.0])

    def _create_diamond_formation(self):
        """Create a diamond formation"""
        if len(self.robot_ids) >= 4:
            positions = [
                np.array([0.0, 1.0]),    # Top
                np.array([-1.0, 0.0]),   # Left
                np.array([1.0, 0.0]),    # Right
                np.array([0.0, -1.0])    # Bottom
            ]

            for i, robot_id in enumerate(self.robot_ids):
                if i < 4:
                    self.desired_formation[robot_id] = positions[i]
                else:
                    # For more than 4 robots, create multiple diamonds
                    self.desired_formation[robot_id] = positions[i % 4] + np.array([i//4 * 3, 0])

    def _create_circle_formation(self):
        """Create a circular formation"""
        n_robots = len(self.robot_ids)
        radius = 2.0

        for i, robot_id in enumerate(self.robot_ids):
            angle = 2 * math.pi * i / n_robots
            self.desired_formation[robot_id] = np.array([
                radius * math.cos(angle),
                radius * math.sin(angle)
            ])

    def _create_v_formation(self):
        """Create a V formation"""
        n_robots = len(self.robot_ids)

        # Leader at the front
        if n_robots > 0:
            self.desired_formation[self.robot_ids[0]] = np.array([0.0, 0.0])

        # Others in V shape
        for i in range(1, n_robots):
            side = (-1) ** i  # Alternates between -1 and 1
            row = (i + 1) // 2
            self.desired_formation[self.robot_ids[i]] = np.array([
                row * 0.8,  # x: back from leader
                side * row * 0.6  # y: offset in V shape
            ])

    def _create_grid_formation(self):
        """Create a grid formation"""
        n_robots = len(self.robot_ids)
        grid_size = int(math.ceil(math.sqrt(n_robots)))

        for i, robot_id in enumerate(self.robot_ids):
            row = i // grid_size
            col = i % grid_size
            self.desired_formation[robot_id] = np.array([col * 1.0, row * 1.0])

    def set_neighbors(self, robot_id: str, neighbor_ids: List[str]):
        """Set neighbors for a robot"""
        self.neighbors[robot_id] = neighbor_ids

    def update_formation(self, dt: float = 0.1):
        """Update robot positions to maintain formation"""
        if not self.formation_active:
            return

        for robot_id in self.robot_ids:
            current_pos = self.robot_positions[robot_id]
            desired_pos = self.desired_formation[robot_id]

            # Formation control: move towards desired position
            formation_error = desired_pos - current_pos

            # Local coordination: consider neighbors
            coordination_force = np.array([0.0, 0.0])
            for neighbor_id in self.neighbors[robot_id]:
                if neighbor_id in self.robot_positions:
                    neighbor_pos = self.robot_positions[neighbor_id]
                    # Maintain desired distance from neighbors
                    desired_distance = 1.0
                    actual_distance = np.linalg.norm(current_pos - neighbor_pos)
                    direction = current_pos - neighbor_pos
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction)

                    if actual_distance < desired_distance * 0.8:  # Too close
                        coordination_force += direction * (desired_distance - actual_distance) * 0.5
                    elif actual_distance > desired_distance * 1.2:  # Too far
                        coordination_force -= direction * (actual_distance - desired_distance) * 0.5

            # Combine formation and coordination forces
            total_force = formation_error * 0.8 + coordination_force * 0.2

            # Update velocity (with damping)
            self.robot_velocities[robot_id] = (0.7 * self.robot_velocities[robot_id] +
                                             0.3 * total_force)

            # Update position
            self.robot_positions[robot_id] += self.robot_velocities[robot_id] * dt

    def move_formation(self, offset: np.ndarray):
        """Move the entire formation by an offset"""
        for robot_id in self.robot_ids:
            if robot_id in self.desired_formation:
                self.desired_formation[robot_id] += offset

    def get_robot_positions(self) -> Dict[str, np.ndarray]:
        """Get current robot positions"""
        return self.robot_positions.copy()

    def get_formation_error(self) -> float:
        """Calculate total formation error"""
        total_error = 0.0
        for robot_id in self.robot_ids:
            current_pos = self.robot_positions[robot_id]
            desired_pos = self.desired_formation[robot_id]
            error = np.linalg.norm(current_pos - desired_pos)
            total_error += error
        return total_error / len(self.robot_ids)

class DistributedFormationController:
    def __init__(self, robot_ids: List[str], comm_range: float = 2.0):
        self.robot_ids = robot_ids
        self.comm_range = comm_range
        self.robots = {}

        # Initialize each robot's local controller
        for robot_id in robot_ids:
            self.robots[robot_id] = {
                'position': np.array([0.0, 0.0]),
                'velocity': np.array([0.0, 0.0]),
                'neighbors': [],
                'desired_position': np.array([0.0, 0.0]),
                'formation_offset': np.array([0.0, 0.0])
            }

    def update_neighbors(self):
        """Update neighbor lists based on communication range"""
        for robot_id in self.robot_ids:
            robot_pos = self.robots[robot_id]['position']
            neighbors = []

            for other_id in self.robot_ids:
                if other_id != robot_id:
                    other_pos = self.robots[other_id]['position']
                    distance = np.linalg.norm(robot_pos - other_pos)

                    if distance <= self.comm_range:
                        neighbors.append(other_id)

            self.robots[robot_id]['neighbors'] = neighbors

    def set_formation_pattern(self, pattern: str):
        """Set formation pattern for all robots"""
        n_robots = len(self.robot_ids)

        if pattern == "line":
            for i, robot_id in enumerate(self.robot_ids):
                self.robots[robot_id]['desired_position'] = np.array([i, 0])
        elif pattern == "circle":
            radius = 2.0
            for i, robot_id in enumerate(self.robot_ids):
                angle = 2 * math.pi * i / n_robots
                self.robots[robot_id]['desired_position'] = np.array([
                    radius * math.cos(angle),
                    radius * math.sin(angle)
                ])

    def update_robot(self, robot_id: str, dt: float = 0.1):
        """Update a single robot's position"""
        robot = self.robots[robot_id]

        current_pos = robot['position']
        desired_pos = robot['desired_position'] + robot['formation_offset']

        # Formation force: move towards desired position
        formation_force = 0.5 * (desired_pos - current_pos)

        # Coordination force: maintain distance from neighbors
        coord_force = np.array([0.0, 0.0])

        for neighbor_id in robot['neighbors']:
            neighbor_pos = self.robots[neighbor_id]['position']

            # Calculate distance and direction to neighbor
            diff = current_pos - neighbor_pos
            distance = np.linalg.norm(diff)

            if distance < 0.5:  # Too close - repel
                if distance > 0:
                    direction = diff / distance
                    coord_force += direction * (0.5 - distance) * 2.0
            elif distance > 1.5:  # Too far - attract
                if distance > 0:
                    direction = diff / distance
                    coord_force -= direction * (distance - 1.5) * 0.5

        # Combine forces
        total_force = formation_force + coord_force

        # Update velocity and position
        robot['velocity'] = 0.8 * robot['velocity'] + 0.2 * total_force
        robot['position'] += robot['velocity'] * dt

    def update_all_robots(self, dt: float = 0.1):
        """Update all robots in the formation"""
        self.update_neighbors()

        for robot_id in self.robot_ids:
            self.update_robot(robot_id, dt)

    def move_formation(self, offset: np.ndarray):
        """Move the entire formation"""
        for robot_id in self.robot_ids:
            self.robots[robot_id]['formation_offset'] += offset

# Example usage
if __name__ == "__main__":
    print("Testing Formation Control...")

    # Create formation controller
    robot_ids = ['R1', 'R2', 'R3', 'R4', 'R5']
    formation_ctrl = FormationController(robot_ids)

    # Set up neighbors (simple chain topology)
    for i, robot_id in enumerate(robot_ids):
        neighbors = []
        if i > 0:
            neighbors.append(robot_ids[i-1])
        if i < len(robot_ids) - 1:
            neighbors.append(robot_ids[i+1])
        formation_ctrl.set_neighbors(robot_id, neighbors)

    # Set formation
    formation_ctrl.set_formation('line')
    print("Initial formation set to line")

    # Simulate formation maintenance
    for step in range(50):
        formation_ctrl.update_formation(dt=0.1)
        if step % 10 == 0:
            error = formation_ctrl.get_formation_error()
            print(f"Step {step}, Formation error: {error:.4f}")

    final_positions = formation_ctrl.get_robot_positions()
    print("Final positions:", {k: v.tolist() for k, v in final_positions.items()})

    # Test distributed formation controller
    dist_formation = DistributedFormationController(robot_ids, comm_range=3.0)
    dist_formation.set_formation_pattern('circle')

    print("\nTesting Distributed Formation Control...")
    for step in range(30):
        dist_formation.update_all_robots(dt=0.1)
        if step % 10 == 0:
            positions = [dist_formation.robots[rid]['position'] for rid in robot_ids]
            avg_distance = np.mean([np.linalg.norm(pos) for pos in positions])
            print(f"Step {step}, Avg distance from center: {avg_distance:.4f}")
```

### Task Allocation Algorithms

```python
#!/usr/bin/env python3

import numpy as np
from typing import List, Dict, Tuple
import random
from scipy.optimize import linear_sum_assignment

class TaskAllocator:
    def __init__(self, robot_ids: List[str], task_ids: List[str]):
        self.robot_ids = robot_ids
        self.task_ids = task_ids
        self.cost_matrix = np.zeros((len(robot_ids), len(task_ids)))
        self.assignments = {}
        self.unassigned_tasks = set(task_ids)
        self.occupied_robots = set()

    def calculate_cost_matrix(self, robot_capabilities: Dict[str, List[str]],
                            task_requirements: Dict[str, List[str]],
                            robot_positions: Dict[str, Tuple[float, float]],
                            task_locations: Dict[str, Tuple[float, float]]) -> np.ndarray:
        """Calculate cost matrix for task allocation"""
        cost_matrix = np.zeros((len(self.robot_ids), len(self.task_ids)))

        for i, robot_id in enumerate(self.robot_ids):
            for j, task_id in enumerate(self.task_ids):
                # Distance cost
                robot_pos = robot_positions[robot_id]
                task_pos = task_locations[task_id]
                distance = np.sqrt((robot_pos[0] - task_pos[0])**2 +
                                 (robot_pos[1] - task_pos[1])**2)

                # Capability compatibility cost
                robot_caps = set(robot_capabilities[robot_id])
                task_reqs = set(task_requirements[task_id])
                missing_caps = len(task_reqs - robot_caps)
                capability_cost = missing_caps * 100  # High penalty for missing capabilities

                # Battery/energy cost (simplified)
                battery_cost = 0  # In real system, this would come from robot status

                total_cost = distance + capability_cost + battery_cost
                cost_matrix[i][j] = total_cost

        return cost_matrix

    def allocate_tasks_optimally(self) -> Dict[str, str]:
        """Allocate tasks using Hungarian algorithm for optimal assignment"""
        if len(self.robot_ids) < len(self.task_ids):
            print("Warning: More tasks than robots, some tasks will remain unassigned")

        # Use Hungarian algorithm for optimal assignment
        cost_matrix = self.cost_matrix
        if cost_matrix.shape[0] < cost_matrix.shape[1]:  # More tasks than robots
            # Add dummy robots with high cost
            dummy_rows = cost_matrix.shape[1] - cost_matrix.shape[0]
            dummy_costs = np.ones((dummy_rows, cost_matrix.shape[1])) * 1000
            cost_matrix = np.vstack([cost_matrix, dummy_costs])

        robot_indices, task_indices = linear_sum_assignment(cost_matrix)

        assignments = {}
        for robot_idx, task_idx in zip(robot_indices, task_indices):
            if robot_idx < len(self.robot_ids) and task_idx < len(self.task_ids):
                robot_id = self.robot_ids[robot_idx]
                task_id = self.task_ids[task_idx]
                assignments[task_id] = robot_id

        self.assignments = assignments
        self.occupied_robots = set(assignments.values())
        self.unassigned_tasks = set(self.task_ids) - set(assignments.keys())

        return assignments

class AuctionBasedTaskAllocator:
    def __init__(self, robot_ids: List[str], task_ids: List[str]):
        self.robot_ids = robot_ids
        self.task_ids = task_ids
        self.task_bids = {task_id: {} for task_id in task_ids}  # {task_id: {robot_id: bid_value}}
        self.assignments = {}
        self.task_values = {task_id: random.uniform(10, 100) for task_id in task_ids}

    def calculate_bid(self, robot_id: str, task_id: str,
                     robot_capabilities: Dict[str, List[str]],
                     task_requirements: Dict[str, List[str]],
                     robot_positions: Dict[str, Tuple[float, float]],
                     task_locations: Dict[str, Tuple[float, float]]) -> float:
        """Calculate bid value for a robot-task pair"""
        # Task value minus cost of execution
        task_value = self.task_values[task_id]

        # Calculate cost factors
        robot_pos = robot_positions[robot_id]
        task_pos = task_locations[task_id]
        distance = np.sqrt((robot_pos[0] - task_pos[0])**2 +
                          (robot_pos[1] - task_pos[1])**2)

        # Capability match factor (0-1, where 1 is perfect match)
        robot_caps = set(robot_capabilities[robot_id])
        task_reqs = set(task_requirements[task_id])
        capability_match = len(robot_caps & task_reqs) / len(task_reqs) if task_reqs else 1.0

        # Cost is distance divided by capability match (higher match = lower effective cost)
        effective_cost = distance / (capability_match + 0.1)  # +0.1 to avoid division by zero

        # Bid is value minus cost (with capability adjustment)
        bid = task_value - effective_cost * (2 - capability_match)  # Better capability = higher bid

        return max(0, bid)  # Bids cannot be negative

    def run_auction(self, robot_capabilities: Dict[str, List[str]],
                   task_requirements: Dict[str, List[str]],
                   robot_positions: Dict[str, Tuple[float, float]],
                   task_locations: Dict[str, Tuple[float, float]]) -> Dict[str, str]:
        """Run task allocation auction"""
        # Initialize bids
        for task_id in self.task_ids:
            for robot_id in self.robot_ids:
                bid = self.calculate_bid(robot_id, task_id, robot_capabilities,
                                       task_requirements, robot_positions, task_locations)
                self.task_bids[task_id][robot_id] = bid

        # Assign tasks to highest bidders
        assignments = {}
        assigned_robots = set()

        # Sort tasks by some priority (here, by value)
        sorted_tasks = sorted(self.task_ids, key=lambda t: self.task_values[t], reverse=True)

        for task_id in sorted_tasks:
            # Find highest bidding robot that hasn't been assigned
            available_bids = {robot_id: bid for robot_id, bid in self.task_bids[task_id].items()
                            if robot_id not in assigned_robots and bid > 0}

            if available_bids:
                winner = max(available_bids.keys(), key=lambda r: available_bids[r])
                assignments[task_id] = winner
                assigned_robots.add(winner)

        self.assignments = assignments
        return assignments

class MarketBasedTaskAllocator:
    def __init__(self, robot_ids: List[str], task_ids: List[str]):
        self.robot_ids = robot_ids
        self.task_ids = task_ids
        self.supply_demand = {}
        self.price_history = {}
        self.assignments = {}

    def initialize_market(self):
        """Initialize market prices for tasks"""
        for task_id in self.task_ids:
            self.price_history[task_id] = [10.0]  # Start with base price

    def calculate_robot_utility(self, robot_id: str, task_id: str,
                              robot_capabilities: Dict[str, List[str]],
                              task_requirements: Dict[str, List[str]],
                              robot_positions: Dict[str, Tuple[float, float]],
                              task_locations: Dict[str, Tuple[float, float]],
                              current_price: float) -> float:
        """Calculate utility for a robot performing a task"""
        # Calculate benefit (value) minus cost
        robot_pos = robot_positions[robot_id]
        task_pos = task_locations[task_id]
        distance = np.sqrt((robot_pos[0] - task_pos[0])**2 +
                          (robot_pos[1] - task_pos[1])**2)

        # Capability match factor
        robot_caps = set(robot_capabilities[robot_id])
        task_reqs = set(task_requirements[task_id])
        capability_match = len(robot_caps & task_reqs) / len(task_reqs) if task_reqs else 1.0

        # Base value of task (this could come from task priority, etc.)
        base_value = 50.0

        # Utility = (base_value + price) * capability_match - distance_cost
        utility = (base_value + current_price) * capability_match - distance

        return utility

    def update_market_prices(self):
        """Update prices based on supply and demand"""
        for task_id in self.task_ids:
            current_price = self.price_history[task_id][-1]

            # In a real market, this would be based on supply/demand
            # For simulation, we'll adjust based on assignment success
            if task_id in self.assignments:
                # Task was assigned, price could decrease
                new_price = max(1.0, current_price * 0.95)
            else:
                # Task not assigned, price should increase
                new_price = min(100.0, current_price * 1.05)

            self.price_history[task_id].append(new_price)

    def allocate_tasks_market(self, robot_capabilities: Dict[str, List[str]],
                            task_requirements: Dict[str, List[str]],
                            robot_positions: Dict[str, Tuple[float, float]],
                            task_locations: Dict[str, Tuple[float, float]]) -> Dict[str, str]:
        """Allocate tasks using market-based approach"""
        self.initialize_market()

        # Run multiple market rounds
        for round_num in range(10):  # 10 market rounds
            # Calculate utilities for all robot-task pairs
            utilities = {}
            for task_id in self.task_ids:
                current_price = self.price_history[task_id][-1]
                for robot_id in self.robot_ids:
                    utility = self.calculate_robot_utility(
                        robot_id, task_id, robot_capabilities, task_requirements,
                        robot_positions, task_locations, current_price
                    )
                    if robot_id not in utilities:
                        utilities[robot_id] = {}
                    utilities[robot_id][task_id] = utility

            # Each robot chooses its most preferred task
            robot_choices = {}
            for robot_id in self.robot_ids:
                if robot_id in utilities:
                    # Choose task with highest utility
                    best_task = max(utilities[robot_id].keys(),
                                  key=lambda t: utilities[robot_id][t])
                    robot_choices[robot_id] = best_task

            # Resolve conflicts (multiple robots choosing same task)
            task_assignments = {}
            for robot_id, task_id in robot_choices.items():
                if task_id not in task_assignments:
                    task_assignments[task_id] = robot_id
                else:
                    # Conflict: choose based on utility
                    current_robot = task_assignments[task_id]
                    current_utility = utilities[current_robot][task_id]
                    challenger_utility = utilities[robot_id][task_id]

                    if challenger_utility > current_utility:
                        task_assignments[task_id] = robot_id

            self.assignments = task_assignments
            self.update_market_prices()

        return self.assignments

# Example usage
if __name__ == "__main__":
    print("Testing Task Allocation Algorithms...")

    # Define robots and tasks
    robot_ids = ['R1', 'R2', 'R3', 'R4']
    task_ids = ['T1', 'T2', 'T3', 'T4']

    # Define capabilities and requirements
    robot_capabilities = {
        'R1': ['navigation', 'manipulation'],
        'R2': ['navigation', 'sensing'],
        'R3': ['navigation', 'manipulation', 'sensing'],
        'R4': ['navigation', 'communication']
    }

    task_requirements = {
        'T1': ['navigation', 'manipulation'],
        'T2': ['navigation', 'sensing'],
        'T3': ['navigation', 'manipulation', 'sensing'],
        'T4': ['navigation', 'communication']
    }

    # Define positions
    robot_positions = {
        'R1': (0, 0),
        'R2': (5, 0),
        'R3': (0, 5),
        'R4': (5, 5)
    }

    task_locations = {
        'T1': (1, 1),
        'T2': (4, 1),
        'T3': (1, 4),
        'T4': (4, 4)
    }

    # Test optimal allocation
    allocator = TaskAllocator(robot_ids, task_ids)
    cost_matrix = allocator.calculate_cost_matrix(
        robot_capabilities, task_requirements, robot_positions, task_locations
    )
    allocator.cost_matrix = cost_matrix
    optimal_assignments = allocator.allocate_tasks_optimally()
    print("Optimal assignments:", optimal_assignments)

    # Test auction-based allocation
    auction_allocator = AuctionBasedTaskAllocator(robot_ids, task_ids)
    auction_assignments = auction_allocator.run_auction(
        robot_capabilities, task_requirements, robot_positions, task_locations
    )
    print("Auction assignments:", auction_assignments)

    # Test market-based allocation
    market_allocator = MarketBasedTaskAllocator(robot_ids, task_ids)
    market_assignments = market_allocator.allocate_tasks_market(
        robot_capabilities, task_requirements, robot_positions, task_locations
    )
    print("Market assignments:", market_assignments)
```

## Practical Exercises

### Exercise 1: Implement a Multi-Robot Communication System

**Objective**: Create a robust communication system that allows multiple robots to exchange information reliably.

**Steps**:
1. Implement a message passing system with broadcast and unicast capabilities
2. Add message serialization and deserialization
3. Implement communication reliability mechanisms (acknowledgments, retries)
4. Test with various network topologies
5. Evaluate communication performance under different conditions

**Expected Outcome**: A working multi-robot communication system that can handle message passing with reliability guarantees.

### Exercise 2: Design a Formation Control System

**Objective**: Create a formation control system that can maintain geometric patterns with multiple robots.

**Steps**:
1. Implement basic formation patterns (line, circle, V-formation)
2. Add distributed formation control algorithms
3. Implement collision avoidance between robots
4. Test with dynamic reconfiguration
5. Evaluate formation stability and robustness

**Expected Outcome**: A formation control system that can maintain stable formations with multiple robots while avoiding collisions.

### Exercise 3: Task Allocation Competition

**Objective**: Compare different task allocation algorithms in a multi-robot system.

**Steps**:
1. Implement multiple allocation algorithms (auction, market-based, centralized)
2. Create a simulation environment with tasks and robots
3. Compare algorithms based on efficiency, fairness, and scalability
4. Test with different task distributions and robot capabilities
5. Analyze performance under various conditions

**Expected Outcome**: A comparison of different task allocation approaches with performance analysis.

## Chapter Summary

This chapter covered the essential concepts of multi-robot systems and coordination:

1. **Communication Systems**: Implementing reliable communication networks for multi-robot coordination.

2. **Consensus Algorithms**: Distributed algorithms that allow robots to agree on values or decisions.

3. **Coordination Strategies**: Market-based, auction-based, and other coordination mechanisms.

4. **Swarm Intelligence**: Bio-inspired algorithms for collective robot behavior.

5. **Formation Control**: Algorithms for maintaining geometric patterns with multiple robots.

6. **Task Allocation**: Methods for assigning tasks to robots in a coordinated manner.

Multi-robot systems leverage the collective capabilities of multiple agents to achieve goals that would be difficult for single robots. Success in multi-robot systems requires careful design of communication protocols, coordination mechanisms, and task allocation strategies that balance optimality with scalability and robustness.

## Further Reading

1. "Multi-Robot Systems: From Swarms to Intelligent Automata" by Parker - Comprehensive overview of multi-robot systems
2. "Swarm Intelligence: From Natural to Artificial Systems" by Bonabeau et al. - Foundational swarm intelligence text
3. "Distributed Algorithms" by Lynch - Theoretical foundations for distributed coordination
4. "Introduction to Multi-Agent Systems" by Wooldridge - Agent-based coordination approaches
5. "Cooperative Control of Dynamical Systems" by Lewis - Control-theoretic approaches to coordination

## Assessment Questions

1. Compare centralized vs. decentralized coordination approaches in multi-robot systems.

2. Implement a consensus algorithm for multi-robot average computation.

3. Design a communication protocol for a team of 10 robots with limited range.

4. Analyze the scalability of different task allocation algorithms.

5. Implement a formation control system for maintaining a circular pattern.

6. Discuss the trade-offs between optimality and computational complexity in multi-robot coordination.

7. Design a fault-tolerant coordination system that handles robot failures.

8. Compare auction-based and market-based task allocation mechanisms.

9. Evaluate the impact of communication delays on multi-robot system performance.

10. Design a swarm intelligence algorithm for area coverage with multiple robots.

