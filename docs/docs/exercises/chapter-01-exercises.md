---
sidebar_position: 1
---

# Chapter 1 Exercises: Introduction to Physical AI and Humanoid Robotics

## Exercise 1: Kinematics Analysis

### Problem
Consider a simple 2-DOF planar robot leg with:
- Hip joint at origin (0, 0)
- Thigh length: 0.4 m
- Shank length: 0.4 m
- Joint angles: q1 = 45°, q2 = -30°

1. Calculate the position of the ankle using forward kinematics
2. Verify your result using a geometric approach
3. Plot the leg configuration using Python

### Solution Template
```python
import numpy as np
import matplotlib.pyplot as plt

# Given parameters
q1 = np.radians(45)  # Hip angle in radians
q2 = np.radians(-30) # Knee angle in radians
l1 = 0.4  # Thigh length (m)
l2 = 0.4  # Shank length (m)

# Calculate forward kinematics
# TODO: Implement forward kinematics equations
ankle_x = # Your calculation here
ankle_y = # Your calculation here

print(f"Ankle position: ({ankle_x:.3f}, {ankle_y:.3f})")

# Visualization
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the leg
hip = [0, 0]
knee = [l1 * np.cos(q1), l1 * np.sin(q1)]
ankle = [ankle_x, ankle_y]

ax.plot([hip[0], knee[0]], [hip[1], knee[1]], 'b-', linewidth=3, label='Thigh')
ax.plot([knee[0], ankle[0]], [knee[1], ankle[1]], 'r-', linewidth=3, label='Shank')

ax.plot(hip[0], hip[1], 'go', markersize=10, label='Hip')
ax.plot(knee[0], knee[1], 'yo', markersize=10, label='Knee')
ax.plot(ankle[0], ankle[1], 'ro', markersize=10, label='Ankle')

ax.set_xlim(-0.2, 0.8)
ax.set_ylim(-0.2, 0.8)
ax.set_aspect('equal')
ax.grid(True)
ax.legend()
ax.set_title('Robot Leg Configuration')

plt.show()
```

## Exercise 2: Center of Mass Calculation

### Problem
A simple humanoid model has the following components:
- Torso: mass = 20kg, position = (0, 0.8, 0)
- Head: mass = 5kg, position = (0, 1.6, 0)
- Right arm: mass = 3kg, position = (0.3, 1.2, 0)
- Left arm: mass = 3kg, position = (-0.3, 1.2, 0)
- Right leg: mass = 10kg, position = (0.1, 0.2, 0)
- Left leg: mass = 10kg, position = (-0.1, 0.2, 0)

Calculate the overall center of mass of this humanoid.

### Solution Template
```python
import numpy as np

def calculate_com(positions, masses):
    # TODO: Implement center of mass calculation
    pass

# Given data
positions = [
    [0, 0.8, 0],    # Torso
    [0, 1.6, 0],    # Head
    [0.3, 1.2, 0],  # Right arm
    [-0.3, 1.2, 0], # Left arm
    [0.1, 0.2, 0],  # Right leg
    [-0.1, 0.2, 0]  # Left leg
]

masses = [20, 5, 3, 3, 10, 10]  # in kg

# Calculate center of mass
com = calculate_com(positions, masses)
print(f"Center of mass: ({com[0]:.3f}, {com[1]:.3f}, {com[2]:.3f})")
```

## Exercise 3: Stability Analysis

### Problem
A humanoid robot has a foot base of support defined by the rectangle with corners at:
- (-0.1, -0.05), (0.1, -0.05), (0.1, 0.05), (-0.1, 0.05)

The center of mass is currently at (0.05, 0.1, 0). Is the robot stable? Why or why not?

### Questions to Consider:
1. What is the margin of stability in the x-direction?
2. What would happen if the CoM shifted to (0.15, 0.1, 0)?
3. How would you modify the robot's pose to increase stability?

## Exercise 4: ROS2 Node Implementation

### Problem
Extend the balance controller example from the chapter to include:
1. PID control instead of simple proportional control
2. Safety limits on joint commands
3. Logging of CoM trajectory

### Implementation Tasks:
1. Add PID controller parameters (Kp, Ki, Kd)
2. Implement integral and derivative terms
3. Add bounds checking for joint commands
4. Create a launch file for the controller

## Challenge Exercise: Walking Pattern Generation

### Problem
Design a simple walking pattern generator that creates a sequence of CoM positions to achieve forward locomotion.

### Requirements:
1. Generate a trajectory for 10 steps
2. Each step should be 0.3m forward
3. Include lateral stability (CoM should move in a zig-zag pattern)
4. Plot the resulting CoM trajectory

### Hints:
- Consider the inverted pendulum model
- Think about foot placement relative to CoM
- Remember that stability requires CoM to be within the support polygon