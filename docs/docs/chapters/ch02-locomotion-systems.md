---
id: ch02-locomotion-systems
title: Locomotion Systems
sidebar_label: "Chapter 2: Locomotion Systems"
sidebar_position: 2
description: Understanding different types of locomotion in humanoid robots
---

# Chapter 2: Locomotion Systems

## Learning Objectives

By the end of this chapter, students will be able to:
- Understand different types of locomotion in humanoid robots
- Analyze the mechanics of bipedal walking
- Implement basic walking pattern generators
- Explain the role of balance control in locomotion
- Design controllers for stable locomotion

## Theoretical Foundations

### Types of Locomotion

Humanoid robots can achieve locomotion through various mechanisms:

1. **Static Walking**: Maintains stability at every step by keeping the center of pressure (CoP) within the support polygon
2. **Dynamic Walking**: Uses dynamic balance where the robot is only stable during motion
3. **Running**: Involves flight phases where both feet are off the ground
4. **Climbing**: Specialized locomotion for navigating obstacles and stairs

### Bipedal Walking Mechanics

Bipedal walking in humanoid robots mimics human walking but with important differences:

- **Double Support Phase**: Both feet on the ground, providing maximum stability
- **Single Support Phase**: One foot on the ground, requiring dynamic balance
- **Impact Phase**: When the swing foot contacts the ground
- **Pre-swing Phase**: Preparation for the next step

### Inverted Pendulum Model

The inverted pendulum is a fundamental model for understanding bipedal balance:

- The robot's body is treated as a point mass on top of a massless leg
- The equation of motion is: θ'' = (g/L) * sin(θ) - (u/mL²) * cos(θ)
- Where θ is the tilt angle, g is gravity, L is leg length, u is control input

## Practical Examples

### Example 1: ZMP (Zero Moment Point) Calculation

```python
import numpy as np

def calculate_zmp(com_pos, com_vel, com_acc, gravity=9.81, height=0.8):
    """
    Calculate Zero Moment Point for bipedal robot
    com_pos: Center of mass position [x, y, z]
    com_vel: Center of mass velocity [vx, vy, vz]
    com_acc: Center of mass acceleration [ax, ay, az]
    """
    x, y, z = com_pos
    vx, vy, vz = com_vel
    ax, ay, az = com_acc

    # ZMP equations (simplified for horizontal movement)
    zmp_x = x - height * ax / (gravity + az)
    zmp_y = y - height * ay / (gravity + az)

    return zmp_x, zmp_y

# Example usage
com_pos = [0.0, 0.0, 0.8]  # CoM at origin, 0.8m high
com_vel = [0.1, 0.0, 0.0]  # Moving forward at 0.1 m/s
com_acc = [0.5, 0.0, 0.0]  # Accelerating forward

zmp = calculate_zmp(com_pos, com_vel, com_acc)
print(f"ZMP position: ({zmp[0]:.3f}, {zmp[1]:.3f})")
```

### Example 2: Simple Walking Pattern Generator

```python
import numpy as np
import matplotlib.pyplot as plt

class WalkingPatternGenerator:
    def __init__(self, step_length=0.3, step_height=0.05, step_time=1.0):
        self.step_length = step_length
        self.step_height = step_height
        self.step_time = step_time
        self.omega = np.sqrt(9.81 / 0.8)  # sqrt(g/h) for inverted pendulum

    def generate_foot_trajectory(self, step_num, support_leg='left'):
        """
        Generate foot trajectory for a single step
        """
        t = np.linspace(0, self.step_time, int(self.step_time * 100))  # 100 Hz

        # Horizontal movement (cubic polynomial)
        x_start = step_num * self.step_length
        x_end = (step_num + 1) * self.step_length
        x = x_start + (x_end - x_start) * (3*(t/self.step_time)**2 - 2*(t/self.step_time)**3)

        # Vertical movement (cubic polynomial)
        y = np.zeros_like(t)
        if support_leg == 'left':
            y = self.step_height * np.sin(np.pi * t / self.step_time)**2
        else:
            y = -self.step_height * np.sin(np.pi * t / self.step_time)**2

        return t, x, y

    def generate_com_trajectory(self, steps):
        """
        Generate CoM trajectory using inverted pendulum model
        """
        t_total = steps * self.step_time
        t = np.linspace(0, t_total, int(t_total * 100))

        # Simple CoM trajectory (simplified)
        com_x = np.zeros_like(t)
        com_y = np.zeros_like(t)

        for i in range(steps):
            start_t = i * self.step_time
            end_t = (i + 1) * self.step_time

            mask = (t >= start_t) & (t < end_t)
            if i % 2 == 0:  # Left foot support
                com_y[mask] = 0.1 * np.sin(np.pi * (t[mask] - start_t) / self.step_time)
            else:  # Right foot support
                com_y[mask] = -0.1 * np.sin(np.pi * (t[mask] - start_t) / self.step_time)

        com_x = np.linspace(0, steps * self.step_length, len(t))

        return t, com_x, com_y

# Example usage
walker = WalkingPatternGenerator()
t, x, y = walker.generate_foot_trajectory(0, 'left')
com_t, com_x, com_y = walker.generate_com_trajectory(2)

# Plot trajectories
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(t, x, label='Foot X position')
ax1.plot(com_t, com_x, label='CoM X position', linestyle='--')
ax1.set_ylabel('X Position (m)')
ax1.legend()
ax1.grid(True)

ax2.plot(t, y, label='Foot Y position')
ax2.plot(com_t, com_y, label='CoM Y position', linestyle='--')
ax2.set_ylabel('Y Position (m)')
ax2.set_xlabel('Time (s)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
```

## ROS2 Integration Example

Here's a ROS2 node for locomotion control:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState
import numpy as np

class LocomotionController(Node):
    def __init__(self):
        super().__init__('locomotion_controller')

        # Publishers
        self.trajectory_publisher = self.create_publisher(
            Float64MultiArray,
            '/trajectory_commands',
            10
        )

        self.com_publisher = self.create_publisher(
            Point,
            '/desired_com',
            10
        )

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Timer for control loop
        self.timer = self.create_timer(0.01, self.control_loop)

        # Walking parameters
        self.step_length = 0.3  # meters
        self.step_height = 0.05  # meters
        self.step_time = 1.0  # seconds
        self.current_step = 0
        self.current_time = 0.0

        # Initialize joint states
        self.joint_positions = {}

    def joint_state_callback(self, msg):
        for name, pos in zip(msg.name, msg.position):
            self.joint_positions[name] = pos

    def calculate_foot_trajectory(self, time_in_step, step_num, leg='left'):
        """Calculate desired foot trajectory for current step"""
        # Normalize time within step
        t_norm = time_in_step / self.step_time

        # Horizontal trajectory
        x_start = step_num * self.step_length
        x_end = (step_num + 1) * self.step_length
        x = x_start + (x_end - x_start) * (3*t_norm**2 - 2*t_norm**3)

        # Vertical trajectory
        y = 0.0
        if leg == 'right':
            y = self.step_height * np.sin(np.pi * t_norm)**2

        return x, y, 0.0  # z is 0 for now

    def control_loop(self):
        # Update timing
        self.current_time += 0.01  # 100Hz control
        self.current_step = int(self.current_time / self.step_time)
        time_in_step = self.current_time % self.step_time

        # Calculate desired trajectories
        if self.current_step % 2 == 0:  # Left foot swing
            foot_x, foot_y, foot_z = self.calculate_foot_trajectory(
                time_in_step, self.current_step // 2, 'left'
            )
        else:  # Right foot swing
            foot_x, foot_y, foot_z = self.calculate_foot_trajectory(
                time_in_step, self.current_step // 2, 'right'
            )

        # Publish foot trajectory
        foot_traj = Float64MultiArray()
        foot_traj.data = [foot_x, foot_y, foot_z]
        self.trajectory_publisher.publish(foot_traj)

        # Calculate and publish desired CoM
        com_msg = Point()
        com_msg.x = self.current_step * self.step_length  # Rough CoM following
        com_msg.y = 0.1 * np.sin(np.pi * time_in_step / self.step_time)  # Lateral sway
        com_msg.z = 0.8  # Constant height
        self.com_publisher.publish(com_msg)

def main(args=None):
    rclpy.init(args=args)
    controller = LocomotionController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Interactive Exercises

### Exercise 1: ZMP Stability Analysis
1. Implement a ZMP tracking controller
2. Analyze the stability of different walking patterns
3. Determine the stability margin for various CoM positions

### Exercise 2: Walking Gait Optimization
1. Modify the walking pattern generator to minimize energy consumption
2. Implement different step timing strategies
3. Compare the stability of different gait patterns

### Exercise 3: Multi-terrain Walking
1. Adapt the walking controller for different ground conditions
2. Implement slope walking capabilities
3. Test the controller on uneven terrain

## Summary

This chapter explored the fundamental concepts of locomotion in humanoid robots. We covered:

- Different types of locomotion and their characteristics
- The mechanics of bipedal walking and the inverted pendulum model
- ZMP-based stability analysis
- Practical implementation of walking pattern generators
- ROS2 integration for locomotion control

Understanding locomotion is crucial for humanoid robots, as it enables them to navigate complex environments and interact with the world. The next chapter will delve into perception systems that allow robots to understand their environment.

## References and Further Reading

1. Kajita, S. (2005). Humanoid Robotics.
2. Pratt, J., & Walking, I. C. (2006). Virtual Model Control of a Biped Robot.
3. Hof, A. L. (2007). Scaling Bipedal Locomotion.