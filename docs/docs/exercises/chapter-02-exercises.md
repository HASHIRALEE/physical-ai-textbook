---
sidebar_position: 2
---

# Chapter 2 Exercises: Locomotion Systems

## Exercise 1: ZMP Calculation and Analysis

### Problem
A humanoid robot has the following CoM parameters:
- Position: [0.1, 0.0, 0.8] m
- Velocity: [0.2, 0.0, 0.0] m/s
- Acceleration: [1.0, 0.0, 0.0] m/s²

1. Calculate the ZMP position
2. If the robot's foot is a rectangle from (-0.1, -0.05) to (0.1, 0.05), is the ZMP within the support polygon?
3. What would happen to the ZMP if the acceleration was increased to 2.0 m/s²?

### Solution Template
```python
import numpy as np

def calculate_zmp(com_pos, com_vel, com_acc, gravity=9.81):
    """
    Calculate Zero Moment Point
    """
    x, y, z = com_pos
    vx, vy, vz = com_vel
    ax, ay, az = com_acc

    # Calculate ZMP
    zmp_x = x - z * ax / (gravity + az)
    zmp_y = y - z * ay / (gravity + az)

    return zmp_x, zmp_y

# Given parameters
com_pos = [0.1, 0.0, 0.8]
com_vel = [0.2, 0.0, 0.0]
com_acc = [1.0, 0.0, 0.0]

zmp = calculate_zmp(com_pos, com_vel, com_acc)
print(f"ZMP: ({zmp[0]:.3f}, {zmp[1]:.3f})")

# Check if ZMP is within support polygon
foot_min_x, foot_max_x = -0.1, 0.1
foot_min_y, foot_max_y = -0.05, 0.05

within_support = (foot_min_x <= zmp[0] <= foot_max_x and
                  foot_min_y <= zmp[1] <= foot_max_y)

print(f"ZMP within support polygon: {within_support}")
```

## Exercise 2: Walking Pattern Generation

### Problem
Implement a more sophisticated walking pattern generator that:
1. Creates smooth joint trajectories for hip, knee, and ankle
2. Accounts for double support phases
3. Includes foot rotation for turning motions

### Implementation Tasks:
1. Create a 5th-order polynomial trajectory for each joint
2. Implement turning by gradually changing the walking direction
3. Add double support phase (20% of step time)

### Solution Template
```python
import numpy as np
import matplotlib.pyplot as plt

def quintic_trajectory(t, t_start, t_end, x_start, x_end, dx_start=0, dx_end=0, ddx_start=0, ddx_end=0):
    """
    Generate 5th order polynomial trajectory
    """
    T = t_end - t_start
    t_norm = (t - t_start) / T

    # Coefficients for quintic polynomial
    a0 = x_start
    a1 = dx_start * T
    a2 = ddx_start * T**2 / 2
    a3 = 10 * (x_end - x_start) - 6 * dx_start * T - 4 * dx_end * T - 1.5 * ddx_start * T**2 + 0.5 * ddx_end * T**2
    a4 = -15 * (x_end - x_start) + 8 * dx_start * T + 7 * dx_end * T + 1.5 * ddx_start * T**2 - ddx_end * T**2
    a5 = 6 * (x_end - x_start) - 3 * dx_start * T - 3 * dx_end * T - 0.5 * ddx_start * T**2 + 0.5 * ddx_end * T**2

    traj = (a0 + a1*t_norm + a2*t_norm**2 + a3*t_norm**3 + a4*t_norm**4 + a5*t_norm**5)
    return traj

class AdvancedWalkingGenerator:
    def __init__(self, step_length=0.3, step_time=1.0, double_support_ratio=0.2):
        self.step_length = step_length
        self.step_time = step_time
        self.double_support_time = double_support_ratio * step_time
        self.single_support_time = self.step_time - self.double_support_time

    def generate_step_trajectory(self, step_num, support_leg='left'):
        """
        Generate complete step trajectory including joint angles
        """
        total_time = self.step_time
        t = np.linspace(0, total_time, int(total_time * 200))  # 200 Hz

        # Foot trajectory (simplified)
        foot_x = np.zeros_like(t)
        foot_y = np.zeros_like(t)
        foot_z = np.zeros_like(t)

        # Calculate trajectories for each phase
        for i, time in enumerate(t):
            if time < self.double_support_time / 2:
                # First double support phase
                foot_x[i] = step_num * self.step_length
            elif time < self.double_support_time / 2 + self.single_support_time:
                # Single support phase
                single_t = time - self.double_support_time / 2
                progress = single_t / self.single_support_time
                foot_x[i] = step_num * self.step_length + progress * self.step_length
                # Add foot lift during single support
                if support_leg == 'right':
                    foot_z[i] = 0.05 * np.sin(np.pi * progress)
            else:
                # Second double support phase
                foot_x[i] = (step_num + 1) * self.step_length

        return t, foot_x, foot_y, foot_z

# Test the generator
walker = AdvancedWalkingGenerator()
t, foot_x, foot_y, foot_z = walker.generate_step_trajectory(0, 'right')

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(t, foot_x)
plt.title('Foot X Position')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')

plt.subplot(1, 3, 2)
plt.plot(t, foot_y)
plt.title('Foot Y Position')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')

plt.subplot(1, 3, 3)
plt.plot(t, foot_z)
plt.title('Foot Z Position')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')

plt.tight_layout()
plt.show()
```

## Exercise 3: Balance Controller Integration

### Problem
Integrate the ZMP-based balance controller with the walking pattern generator to create a stable walking controller.

### Tasks:
1. Implement ZMP feedback control
2. Adjust the walking pattern based on balance errors
3. Test the controller with disturbances

### Implementation Hints:
- Use the ZMP error to adjust the next foot placement
- Implement admittance control for balance recovery
- Add sensory feedback from simulated IMU

## Exercise 4: ROS2 Walking Controller

### Problem
Extend the ROS2 walking controller to:
1. Include proper joint control for all leg joints
2. Implement safety checks and limits
3. Add state machine for different walking phases

### Requirements:
1. Create a state machine with states: STANDING, WALKING, STOPPING, EMERGENCY
2. Implement joint position/velocity controllers
3. Add logging and diagnostics

## Challenge Exercise: Stair Climbing

### Problem
Modify the walking controller to handle stair climbing:
1. Detect stairs using simulated sensors
2. Adjust step height and timing for stairs
3. Maintain balance during stair navigation

### Considerations:
- How does the ZMP change during stair climbing?
- What modifications are needed for foot placement?
- How do you handle the different dynamics of stair climbing vs. flat ground walking?