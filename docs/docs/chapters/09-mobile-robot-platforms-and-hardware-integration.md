---
sidebar_position: 9
title: "Chapter 9: Mobile Robot Platforms and Hardware Integration"
---

# Chapter 9: Mobile Robot Platforms and Hardware Integration

## Learning Objectives

By the end of this chapter, students will be able to:
- Understand the design principles and components of mobile robot platforms
- Analyze different types of mobile robot locomotion systems and their applications
- Implement hardware integration techniques for sensors and actuators
- Design communication protocols for robot subsystems
- Evaluate the trade-offs between different mobile robot architectures
- Troubleshoot common hardware integration challenges in robotic systems
- Apply best practices for reliable hardware-software integration
- Design modular hardware architectures for scalable robotic platforms

## Theoretical Foundations

### Mobile Robot Classification and Design Principles

Mobile robots can be classified based on their locomotion method, environment, and application domain. The fundamental design principles of mobile robots revolve around mobility, stability, power efficiency, and task-specific functionality.

The primary classification of mobile robots includes:

**Wheeled Robots**: These robots use wheels for locomotion and are the most common type due to their simplicity, efficiency, and well-understood kinematics. They can be further categorized into differential drive, Ackermann steering, and omnidirectional configurations.

**Legged Robots**: These robots use articulated legs for locomotion, providing superior mobility over rough terrain but with increased mechanical complexity and energy consumption.

**Tracked Robots**: Similar to wheeled robots but using continuous tracks, offering better traction and stability on soft or uneven surfaces.

**Aerial Robots**: Including drones and other flying platforms, these robots offer unique mobility capabilities but face challenges in power consumption and payload capacity.

The design of a mobile robot platform must consider several critical factors:
- **Payload capacity**: The robot must be able to carry its own components plus any additional equipment required for its tasks
- **Terrain adaptability**: The locomotion system must be suitable for the intended operating environment
- **Power efficiency**: Battery life is often a limiting factor in mobile robotics
- **Stability**: The robot must maintain balance during operation and in static conditions
- **Modularity**: The design should allow for easy modification and expansion

### Kinematic Models for Mobile Robots

Understanding the kinematic models of mobile robots is crucial for control and navigation. The kinematic model describes the relationship between the robot's motion and its control inputs without considering forces.

For a differential drive robot, the kinematic model can be expressed as:

```
ẋ = v cos(θ)
ẏ = v sin(θ)
θ̇ = ω
```

Where (x, y) is the robot's position, θ is its orientation, v is the linear velocity, and ω is the angular velocity. The relationship between wheel velocities and robot motion is:

```
v = (r/2) * (ωr + ωl)
ω = (r/L) * (ωr - ωl)
```

Where r is the wheel radius, L is the distance between wheels, ωr is the right wheel angular velocity, and ωl is the left wheel angular velocity.

### Hardware Architecture and Communication Protocols

Modern mobile robots typically employ a distributed architecture with multiple microcontrollers or single-board computers handling different subsystems. Communication between these subsystems is critical for coordinated operation.

Common communication protocols in mobile robotics include:
- **UART/Serial**: Simple point-to-point communication for sensors and actuators
- **I2C**: Multi-drop communication for sensors with standardized addresses
- **SPI**: High-speed communication for sensors requiring fast data transfer
- **CAN Bus**: Robust communication for automotive and industrial applications
- **Ethernet**: High-bandwidth communication for complex systems
- **WiFi/Bluetooth**: Wireless communication for remote control and data transmission

## Locomotion Systems

### Wheeled Locomotion

Wheeled locomotion is the most common form of mobile robot movement due to its efficiency and simplicity. Different wheel configurations offer various advantages:

```python
#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class RobotState:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0  # Orientation in radians
    v: float = 0.0      # Linear velocity
    omega: float = 0.0  # Angular velocity

class DifferentialDriveRobot:
    def __init__(self, wheel_radius: float, wheel_base: float, max_linear_vel: float = 1.0, max_angular_vel: float = np.pi/2):
        """
        Initialize differential drive robot
        :param wheel_radius: Radius of the wheels in meters
        :param wheel_base: Distance between wheels in meters
        :param max_linear_vel: Maximum linear velocity in m/s
        :param max_angular_vel: Maximum angular velocity in rad/s
        """
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        self.state = RobotState()

    def update_kinematics(self, v: float, omega: float, dt: float):
        """
        Update robot state based on differential drive kinematics
        :param v: Linear velocity
        :param omega: Angular velocity
        :param dt: Time step
        """
        # Clamp velocities to maximum values
        v = np.clip(v, -self.max_linear_vel, self.max_linear_vel)
        omega = np.clip(omega, -self.max_angular_vel, self.max_angular_vel)

        # Update state using kinematic model
        self.state.v = v
        self.state.omega = omega

        # Calculate new pose
        if abs(omega) < 1e-6:  # Straight line motion
            self.state.x += v * np.cos(self.state.theta) * dt
            self.state.y += v * np.sin(self.state.theta) * dt
        else:  # Curved motion
            radius = v / omega
            delta_theta = omega * dt
            self.state.x += radius * (np.sin(self.state.theta + delta_theta) - np.sin(self.state.theta))
            self.state.y += radius * (np.cos(self.state.theta) - np.cos(self.state.theta + delta_theta))
            self.state.theta += delta_theta

        # Normalize orientation to [-pi, pi]
        self.state.theta = np.arctan2(np.sin(self.state.theta), np.cos(self.state.theta))

    def inverse_kinematics(self, v: float, omega: float) -> Tuple[float, float]:
        """
        Calculate wheel velocities from desired robot velocities
        :param v: Desired linear velocity
        :param omega: Desired angular velocity
        :return: (left_wheel_vel, right_wheel_vel)
        """
        v_left = v - (omega * self.wheel_base) / 2.0
        v_right = v + (omega * self.wheel_base) / 2.0

        # Convert to wheel angular velocities
        omega_left = v_left / self.wheel_radius
        omega_right = v_right / self.wheel_radius

        return omega_left, omega_right

    def direct_kinematics(self, omega_left: float, omega_right: float) -> Tuple[float, float]:
        """
        Calculate robot velocities from wheel angular velocities
        :param omega_left: Left wheel angular velocity
        :param omega_right: Right wheel angular velocity
        :return: (linear_vel, angular_vel)
        """
        v_left = omega_left * self.wheel_radius
        v_right = omega_right * self.wheel_radius

        v = (v_left + v_right) / 2.0
        omega = (v_right - v_left) / self.wheel_base

        return v, omega

class AckermannDriveRobot:
    def __init__(self, wheelbase: float, track_width: float, max_linear_vel: float = 2.0, max_steering_angle: float = np.pi/4):
        """
        Initialize Ackermann steering robot
        :param wheelbase: Distance between front and rear axles
        :param track_width: Distance between left and right wheels
        :param max_linear_vel: Maximum linear velocity
        :param max_steering_angle: Maximum steering angle
        """
        self.wheelbase = wheelbase
        self.track_width = track_width
        self.max_linear_vel = max_linear_vel
        self.max_steering_angle = max_steering_angle
        self.state = RobotState()

    def update_kinematics(self, v: float, steering_angle: float, dt: float):
        """
        Update robot state based on Ackermann steering kinematics
        :param v: Linear velocity
        :param steering_angle: Steering angle in radians
        :param dt: Time step
        """
        # Clamp inputs
        v = np.clip(v, -self.max_linear_vel, self.max_linear_vel)
        steering_angle = np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle)

        # Calculate turning radius
        if abs(steering_angle) < 1e-6:
            # Straight line motion
            self.state.x += v * np.cos(self.state.theta) * dt
            self.state.y += v * np.sin(self.state.theta) * dt
        else:
            # Curved motion
            radius = self.wheelbase / np.tan(steering_angle)
            omega = v / radius
            delta_theta = omega * dt

            self.state.x += radius * (np.sin(self.state.theta + delta_theta) - np.sin(self.state.theta))
            self.state.y += radius * (np.cos(self.state.theta) - np.cos(self.state.theta + delta_theta))
            self.state.theta += delta_theta

        # Normalize orientation
        self.state.theta = np.arctan2(np.sin(self.state.theta), np.cos(self.state.theta))

class OmnidirectionalRobot:
    def __init__(self, radius: float, max_wheel_vel: float = 3.0):
        """
        Initialize omnidirectional robot (mecanum wheels)
        :param radius: Distance from center to wheels
        :param max_wheel_vel: Maximum wheel velocity
        """
        self.radius = radius
        self.max_wheel_vel = max_wheel_vel
        self.state = RobotState()

    def update_kinematics(self, vx: float, vy: float, omega: float, dt: float):
        """
        Update robot state based on omnidirectional kinematics
        :param vx: Desired velocity in x direction
        :param vy: Desired velocity in y direction
        :param omega: Desired angular velocity
        :param dt: Time step
        """
        # Update position
        self.state.x += vx * dt
        self.state.y += vy * dt
        self.state.theta += omega * dt

        # Calculate wheel velocities (simplified model)
        # For mecanum wheels, the relationship is more complex
        self.state.v = np.sqrt(vx**2 + vy**2)  # Linear speed
        self.state.omega = omega

# Example usage
if __name__ == "__main__":
    # Create a differential drive robot
    robot = DifferentialDriveRobot(wheel_radius=0.1, wheel_base=0.3)

    # Simulate movement
    dt = 0.1
    trajectory = []

    for t in np.arange(0, 10, dt):
        # Command robot to move in a circle
        v = 0.5  # Constant linear velocity
        omega = 0.3  # Constant angular velocity
        robot.update_kinematics(v, omega, dt)
        trajectory.append((robot.state.x, robot.state.y))

    print(f"Robot moved in a circle, final position: ({robot.state.x:.2f}, {robot.state.y:.2f})")
```

### Legged Locomotion

Legged robots offer superior mobility over rough terrain but require complex control algorithms:

```python
#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

class Leg:
    def __init__(self, hip_position: np.ndarray, lengths: List[float]):
        """
        Initialize a robot leg with specified joint lengths
        :param hip_position: Position of the hip joint in robot frame
        :param lengths: List of link lengths [upper_leg, lower_leg]
        """
        self.hip_position = np.array(hip_position)
        self.lengths = lengths
        self.joint_angles = [0.0, 0.0]  # [hip_angle, knee_angle]

    def forward_kinematics(self) -> np.ndarray:
        """
        Calculate foot position from joint angles
        :return: Foot position in robot frame
        """
        l1, l2 = self.lengths
        theta1, theta2 = self.joint_angles

        # Calculate foot position relative to hip
        x_rel = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
        y_rel = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)

        # Transform to robot frame
        foot_pos = self.hip_position + np.array([x_rel, y_rel])
        return foot_pos

    def inverse_kinematics(self, target_pos: np.ndarray) -> bool:
        """
        Calculate joint angles to reach target position
        :param target_pos: Target foot position in robot frame
        :return: True if solution exists, False otherwise
        """
        # Transform to hip frame
        target_rel = target_pos - self.hip_position
        x, y = target_rel

        # Calculate distance from hip to target
        r = np.sqrt(x**2 + y**2)

        l1, l2 = self.lengths

        # Check if target is reachable
        if r > l1 + l2 or r < abs(l1 - l2):
            return False

        # Calculate knee angle using law of cosines
        cos_knee = (l1**2 + l2**2 - r**2) / (2 * l1 * l2)
        knee_angle = np.pi - np.arccos(np.clip(cos_knee, -1, 1))

        # Calculate hip angle
        cos_hip = (l1**2 + r**2 - l2**2) / (2 * l1 * r)
        alpha = np.arccos(np.clip(cos_hip, -1, 1))
        beta = np.arctan2(y, x)
        hip_angle = beta - alpha

        self.joint_angles = [hip_angle, knee_angle]
        return True

class QuadrupedRobot:
    def __init__(self, body_length: float, body_width: float, leg_lengths: List[float]):
        """
        Initialize a quadruped robot
        :param body_length: Length of the robot body
        :param body_width: Width of the robot body
        :param leg_lengths: Lengths of leg segments [upper, lower]
        """
        self.body_length = body_length
        self.body_width = body_width
        self.leg_lengths = leg_lengths

        # Initialize legs at the four corners of the body
        self.legs = [
            Leg([-body_length/2, -body_width/2], leg_lengths),  # Front left
            Leg([body_length/2, -body_width/2], leg_lengths),   # Front right
            Leg([body_length/2, body_width/2], leg_lengths),    # Back right
            Leg([-body_length/2, body_width/2], leg_lengths)    # Back left
        ]

        # Robot state
        self.position = np.array([0.0, 0.0])
        self.orientation = 0.0
        self.body_height = 0.3  # Default body height

    def move_legs_to_positions(self, target_positions: List[np.ndarray]) -> bool:
        """
        Move all legs to specified target positions
        :param target_positions: List of target positions for each leg
        :return: True if all positions are reachable
        """
        success = True
        for leg, target_pos in zip(self.legs, target_positions):
            if not leg.inverse_kinematics(target_pos):
                success = False
        return success

    def generate_gait(self, step_length: float, step_height: float, phase_offset: List[float],
                     num_steps: int) -> List[List[np.ndarray]]:
        """
        Generate gait pattern for walking
        :param step_length: Length of each step
        :param step_height: Height of foot during swing phase
        :param phase_offset: Phase offset for each leg
        :param num_steps: Number of steps to generate
        :return: List of leg positions for each step
        """
        gait_sequence = []

        for step in range(num_steps):
            leg_positions = []
            for i, leg in enumerate(self.legs):
                # Calculate phase for this leg
                phase = (step * 2 * np.pi / num_steps) + phase_offset[i]

                # Simple tripod gait: legs in support phase vs swing phase
                if i in [0, 2]:  # Front left and back right (tripod 1)
                    support_phase = phase
                else:  # Front right and back left (tripod 2)
                    support_phase = phase + np.pi

                # Generate foot trajectory
                if np.sin(support_phase) > 0:  # Support phase (on ground)
                    x = self.position[0] + (i % 2) * step_length - step_length/2
                    y = self.position[1] + ((i // 2) * 2 - 1) * self.body_width/4
                    z = -self.body_height
                else:  # Swing phase (lifting foot)
                    x = self.position[0] + (i % 2) * step_length - step_length/2 + np.cos(support_phase) * step_length/4
                    y = self.position[1] + ((i // 2) * 2 - 1) * self.body_width/4
                    z = -self.body_height + np.abs(np.sin(support_phase)) * step_height

                leg_positions.append(np.array([x, y, z]))

            gait_sequence.append(leg_positions)

        return gait_sequence

class WheeledRobotController:
    def __init__(self, robot: DifferentialDriveRobot):
        """
        Controller for wheeled robot
        :param robot: Differential drive robot instance
        """
        self.robot = robot
        self.kp_linear = 1.0  # Linear velocity proportional gain
        self.kp_angular = 2.0  # Angular velocity proportional gain
        self.max_linear_acc = 0.5  # Maximum linear acceleration
        self.max_angular_acc = 1.0  # Maximum angular acceleration

    def go_to_pose(self, target_x: float, target_y: float, target_theta: float,
                   tolerance: float = 0.1) -> bool:
        """
        Control robot to reach a target pose
        :param target_x: Target x position
        :param target_y: Target y position
        :param target_theta: Target orientation
        :param tolerance: Position tolerance for success
        :return: True if target reached within tolerance
        """
        dt = 0.01
        prev_linear_vel = 0.0
        prev_angular_vel = 0.0

        for step in range(1000):  # Max steps to prevent infinite loop
            # Calculate errors
            dx = target_x - self.robot.state.x
            dy = target_y - self.robot.state.y
            distance = np.sqrt(dx**2 + dy**2)

            # Calculate angle to target
            angle_to_target = np.arctan2(dy, dx)
            angle_error = angle_to_target - self.robot.state.theta
            angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))  # Normalize to [-π, π]

            # If close to target position, align orientation
            if distance < 0.3:
                angular_vel = self.kp_angular * angle_error
                linear_vel = 0.0
            else:
                # Move toward target
                linear_vel = self.kp_linear * distance
                angular_vel = self.kp_angular * angle_error

            # Limit accelerations
            linear_acc = (linear_vel - prev_linear_vel) / dt
            angular_acc = (angular_vel - prev_angular_vel) / dt

            linear_acc = np.clip(linear_acc, -self.max_linear_acc, self.max_linear_acc)
            angular_acc = np.clip(angular_acc, -self.max_angular_acc, self.max_angular_acc)

            linear_vel = prev_linear_vel + linear_acc * dt
            angular_vel = prev_angular_vel + angular_acc * dt

            # Update robot
            self.robot.update_kinematics(linear_vel, angular_vel, dt)

            # Check if target reached
            if distance < tolerance and abs(angle_error) < 0.1:
                return True

            prev_linear_vel = linear_vel
            prev_angular_vel = angular_vel

        return False  # Failed to reach target in time

# Example usage
if __name__ == "__main__":
    # Create a differential drive robot
    robot = DifferentialDriveRobot(wheel_radius=0.05, wheel_base=0.2)
    controller = WheeledRobotController(robot)

    # Try to go to a target position
    success = controller.go_to_pose(1.0, 1.0, 0.0)
    if success:
        print(f"Target reached! Final position: ({robot.state.x:.2f}, {robot.state.y:.2f})")
    else:
        print(f"Failed to reach target. Final position: ({robot.state.x:.2f}, {robot.state.y:.2f})")
```

## Sensor Integration

### IMU Integration and Sensor Fusion

Inertial Measurement Units (IMUs) are critical for robot localization and control:

```python
#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

class IMU:
    def __init__(self, accel_noise: float = 0.01, gyro_noise: float = 0.001,
                 mag_noise: float = 0.1, bias_drift: float = 0.0001):
        """
        Initialize IMU with noise characteristics
        :param accel_noise: Accelerometer noise (m/s²)
        :param gyro_noise: Gyroscope noise (rad/s)
        :param mag_noise: Magnetometer noise (μT)
        :param bias_drift: Bias drift rate
        """
        self.accel_noise = accel_noise
        self.gyro_noise = gyro_noise
        self.mag_noise = mag_noise
        self.bias_drift = bias_drift

        # True values
        self.true_accel = np.array([0.0, 0.0, 9.81])  # Gravity vector
        self.true_gyro = np.array([0.0, 0.0, 0.0])    # No rotation initially
        self.true_mag = np.array([25.0, 0.0, 40.0])   # Approximate magnetic field

        # Biases (start at 0 but will drift)
        self.accel_bias = np.array([0.0, 0.0, 0.0])
        self.gyro_bias = np.array([0.0, 0.0, 0.0])

    def read_sensors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate IMU readings with noise and bias
        :return: (acceleration, angular_velocity, magnetic_field)
        """
        # Add noise and bias to true values
        accel_reading = self.true_accel + self.accel_bias + np.random.normal(0, self.accel_noise, 3)
        gyro_reading = self.true_gyro + self.gyro_bias + np.random.normal(0, self.gyro_noise, 3)
        mag_reading = self.true_mag + np.random.normal(0, self.mag_noise, 3)

        # Update biases with drift
        self.accel_bias += np.random.normal(0, self.bias_drift, 3)
        self.gyro_bias += np.random.normal(0, self.bias_drift, 3)

        return accel_reading, gyro_reading, mag_reading

class ComplementaryFilter:
    def __init__(self, alpha: float = 0.98):
        """
        Initialize complementary filter for attitude estimation
        :param alpha: Filter parameter (higher = more trust in gyro)
        """
        self.alpha = alpha
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z] quaternion
        self.gravity_estimate = np.array([0.0, 0.0, 1.0])  # Gravity in body frame

    def update(self, accel: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        """
        Update orientation estimate using accelerometer and gyroscope
        :param accel: Accelerometer reading
        :param gyro: Gyroscope reading
        :param dt: Time step
        :return: Updated quaternion [w, x, y, z]
        """
        # Normalize accelerometer reading
        accel_norm = accel / np.linalg.norm(accel)

        # Estimate gravity direction from accelerometer
        gravity_from_accel = -accel_norm  # Accelerometer measures opposite of gravity

        # Integrate gyroscope to get orientation change
        gyro_norm = np.linalg.norm(gyro)
        if gyro_norm > 1e-6:  # Avoid division by zero
            axis = gyro / gyro_norm
            angle = gyro_norm * dt
            # Convert axis-angle to quaternion
            s = np.sin(angle / 2)
            w = np.cos(angle / 2)
            dq = np.array([w, s * axis[0], s * axis[1], s * axis[2]])
        else:
            dq = np.array([1.0, 0.0, 0.0, 0.0])  # No rotation

        # Apply rotation to current orientation
        self.orientation = self.quaternion_multiply(dq, self.orientation)
        self.orientation = self.orientation / np.linalg.norm(self.orientation)  # Normalize

        # Get gravity estimate from current orientation
        R = self.quaternion_to_rotation_matrix(self.orientation)
        gravity_from_orientation = R @ np.array([0, 0, 1])

        # Complementary filter: combine accelerometer and orientation estimates
        filtered_gravity = (self.alpha * gravity_from_orientation +
                           (1 - self.alpha) * gravity_from_accel)
        filtered_gravity = filtered_gravity / np.linalg.norm(filtered_gravity)

        # Update orientation based on filtered gravity
        self.gravity_estimate = filtered_gravity

        return self.orientation

    def quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        return np.array([w, x, y, z])

    def quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix"""
        w, x, y, z = q
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])

class ExtendedKalmanFilter:
    def __init__(self):
        """Initialize EKF for sensor fusion"""
        # State: [x, y, z, vx, vy, vz, qw, qx, qy, qz]
        self.state = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=float)
        self.covariance = np.eye(10) * 0.1  # Initial uncertainty

        # Process noise
        self.Q = np.eye(10) * 0.01

        # Measurement noise
        self.R_imu = np.diag([0.01, 0.01, 0.01,  # accel
                             0.001, 0.001, 0.001,  # gyro
                             0.1, 0.1, 0.1])       # mag

    def predict(self, dt: float):
        """Predict next state based on motion model"""
        x, y, z, vx, vy, vz, qw, qx, qy, qz = self.state

        # Predict new position based on velocity
        self.state[0] += vx * dt
        self.state[1] += vy * dt
        self.state[2] += vz * dt

        # Velocity prediction (assuming constant velocity model)
        # For more complex models, we could include acceleration

        # For orientation, we would integrate angular velocity
        # This is a simplified model

        # Update covariance
        F = self.get_jacobian_F(dt)
        self.covariance = F @ self.covariance @ F.T + self.Q

    def update_with_imu(self, accel: np.ndarray, gyro: np.ndarray, mag: np.ndarray):
        """Update state estimate with IMU measurements"""
        # Measurement model: map state to expected measurements
        # For this example, we'll just update orientation based on IMU
        predicted_accel = self.estimate_gravity_in_body_frame()

        # Innovation (measurement residual)
        innovation = np.concatenate([accel, gyro, mag]) - np.concatenate([predicted_accel, gyro, mag])

        # Jacobian of measurement model
        H = self.get_jacobian_H()

        # Innovation covariance
        S = H @ self.covariance @ H.T + self.R_imu

        # Kalman gain
        K = self.covariance @ H.T @ np.linalg.inv(S)

        # Update state
        self.state += K @ innovation

        # Update covariance
        I = np.eye(len(self.state))
        self.covariance = (I - K @ H) @ self.covariance

    def get_jacobian_F(self, dt: float) -> np.ndarray:
        """Get Jacobian of motion model"""
        F = np.eye(10)
        # Position from velocity
        F[0, 3] = dt  # dx/dvx
        F[1, 4] = dt  # dy/dvy
        F[2, 5] = dt  # dz/dvz
        return F

    def get_jacobian_H(self) -> np.ndarray:
        """Get Jacobian of measurement model"""
        # This is a simplified version - in practice, this would be more complex
        H = np.zeros((9, 10))  # 9 measurements (3 accel + 3 gyro + 3 mag), 10 states
        # For now, we'll just map directly for demonstration
        return H

    def estimate_gravity_in_body_frame(self) -> np.ndarray:
        """Estimate what accelerometer should read based on orientation"""
        # Simplified - in real implementation, this would use full rotation matrix
        return np.array([0, 0, 9.81])  # Assuming we're upright

class SensorIntegrationSystem:
    def __init__(self):
        self.imu = IMU()
        self.complementary_filter = ComplementaryFilter(alpha=0.98)
        self.ekf = ExtendedKalmanFilter()
        self.time = 0.0
        self.dt = 0.01

    def run_simulation(self, duration: float = 10.0) -> Tuple[List, List, List]:
        """Run sensor integration simulation"""
        times = []
        orientations = []
        positions = []

        for t in np.arange(0, duration, self.dt):
            # Read sensors
            accel, gyro, mag = self.imu.read_sensors()

            # Update complementary filter
            quat = self.complementary_filter.update(accel, gyro, self.dt)

            # Update EKF
            self.ekf.predict(self.dt)
            self.ekf.update_with_imu(accel, gyro, mag)

            # Store data
            times.append(t)
            orientations.append(quat.copy())
            positions.append(self.ekf.state[:3].copy())

            self.time += self.dt

        return times, orientations, positions

# Example usage
if __name__ == "__main__":
    # Create sensor integration system
    sensor_system = SensorIntegrationSystem()

    # Run simulation
    times, orientations, positions = sensor_system.run_simulation(duration=5.0)

    print(f"Sensor integration simulation completed for {len(times)} time steps")
    print(f"Final orientation: {orientations[-1]}")
    print(f"Final position: {positions[-1]}")
```

### Camera and LiDAR Integration

```python
#!/usr/bin/env python3

import numpy as np
import cv2
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

class Camera:
    def __init__(self, width: int = 640, height: int = 480, fov: float = 60.0):
        """
        Initialize camera model
        :param width: Image width in pixels
        :param height: Image height in pixels
        :param fov: Field of view in degrees
        """
        self.width = width
        self.height = height
        self.fov = fov * np.pi / 180  # Convert to radians
        self.focal_length = width / (2 * np.tan(self.fov / 2))

        # Camera intrinsic matrix
        self.K = np.array([
            [self.focal_length, 0, width / 2],
            [0, self.focal_length, height / 2],
            [0, 0, 1]
        ])

        # Camera position and orientation
        self.position = np.array([0.0, 0.0, 0.0])
        self.orientation = np.array([0.0, 0.0, 0.0])  # Euler angles [roll, pitch, yaw]

    def project_3d_to_2d(self, points_3d: np.ndarray) -> np.ndarray:
        """
        Project 3D points to 2D image coordinates
        :param points_3d: Array of 3D points [N x 3]
        :return: Array of 2D image coordinates [N x 2]
        """
        # Convert to homogeneous coordinates
        points_h = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])

        # Apply camera extrinsic matrix (simplified - just translation for now)
        R = self.euler_to_rotation_matrix(self.orientation)
        t = -R.T @ self.position
        extrinsic = np.hstack([R.T, t.reshape(3, 1)])

        # Project to image plane
        points_cam = extrinsic @ points_h.T
        points_2d = self.K @ points_cam

        # Convert from homogeneous coordinates
        points_2d = points_2d[:2, :] / points_2d[2, :]
        return points_2d.T

    def euler_to_rotation_matrix(self, euler: np.ndarray) -> np.ndarray:
        """Convert Euler angles to rotation matrix"""
        roll, pitch, yaw = euler

        # Rotation around Z (yaw)
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        # Rotation around Y (pitch)
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        # Rotation around X (roll)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

        return Rz @ Ry @ Rx

class LiDAR:
    def __init__(self, max_range: float = 10.0, angular_resolution: float = 1.0,
                 fov: float = 360.0):
        """
        Initialize LiDAR model
        :param max_range: Maximum detection range in meters
        :param angular_resolution: Angular resolution in degrees
        :param fov: Field of view in degrees
        """
        self.max_range = max_range
        self.angular_resolution = angular_resolution
        self.fov = fov
        self.angles = np.arange(0, fov, angular_resolution)

        # Number of beams
        self.num_beams = len(self.angles)

        # LiDAR position and orientation
        self.position = np.array([0.0, 0.0, 0.0])
        self.orientation = np.array([0.0, 0.0, 0.0])

    def scan(self, environment_points: np.ndarray) -> np.ndarray:
        """
        Simulate LiDAR scan
        :param environment_points: Array of 3D points representing environment [N x 3]
        :return: Array of distances for each angle [num_beams]
        """
        # Convert to LiDAR frame
        points_lidar = environment_points - self.position
        R = self.euler_to_rotation_matrix(self.orientation)
        points_lidar = (R.T @ points_lidar.T).T

        # Initialize ranges
        ranges = np.full(self.num_beams, self.max_range)

        # For each angle, find the closest point
        for i, angle in enumerate(self.angles):
            angle_rad = angle * np.pi / 180
            # Define a narrow sector for this beam
            dx = np.cos(angle_rad)
            dy = np.sin(angle_rad)

            # Calculate angles of all points relative to this beam
            point_angles = np.arctan2(points_lidar[:, 1], points_lidar[:, 0])
            angle_diffs = np.abs(point_angles - angle_rad)
            angle_diffs = np.minimum(angle_diffs, 2*np.pi - angle_diffs)

            # Find points within the beam width (approximate)
            beam_width = self.angular_resolution * np.pi / 180
            in_beam = angle_diffs < beam_width / 2

            if np.any(in_beam):
                # Calculate distances for points in this beam
                distances = np.linalg.norm(points_lidar[in_beam, :2], axis=1)
                if len(distances) > 0:
                    min_dist = np.min(distances)
                    ranges[i] = min(min_dist, self.max_range)

        return ranges

    def euler_to_rotation_matrix(self, euler: np.ndarray) -> np.ndarray:
        """Convert Euler angles to rotation matrix"""
        roll, pitch, yaw = euler

        # Simplified for LiDAR (usually only yaw matters)
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        return Rz

    def ranges_to_cartesian(self, ranges: np.ndarray) -> np.ndarray:
        """
        Convert range measurements to Cartesian coordinates
        :param ranges: Array of range measurements
        :return: Array of 3D points [N x 3]
        """
        points = []

        for i, (angle, range_val) in enumerate(zip(self.angles, ranges)):
            if range_val < self.max_range * 0.99:  # Valid measurement
                angle_rad = angle * np.pi / 180
                x = range_val * np.cos(angle_rad)
                y = range_val * np.sin(angle_rad)
                points.append([x, y, 0])  # Assuming 2D scan

        return np.array(points)

class SensorFusion:
    def __init__(self):
        self.camera = Camera()
        self.lidar = LiDAR()
        self.ego_position = np.array([0.0, 0.0, 0.0])
        self.ego_orientation = np.array([0.0, 0.0, 0.0])

    def get_occupancy_grid(self, lidar_ranges: np.ndarray, grid_size: int = 100,
                          grid_resolution: float = 0.1) -> np.ndarray:
        """
        Create occupancy grid from LiDAR data
        :param lidar_ranges: LiDAR range measurements
        :param grid_size: Size of the grid (grid_size x grid_size)
        :param grid_resolution: Resolution of each grid cell in meters
        :return: Occupancy grid
        """
        grid = np.zeros((grid_size, grid_size))
        center = grid_size // 2

        # Convert ranges to Cartesian points
        points_cart = self.lidar.ranges_to_cartesian(lidar_ranges)

        # Mark occupied cells
        for point in points_cart:
            x_idx = int(center + point[0] / grid_resolution)
            y_idx = int(center + point[1] / grid_resolution)

            if 0 <= x_idx < grid_size and 0 <= y_idx < grid_size:
                grid[x_idx, y_idx] = 1.0  # Occupied

        return grid

    def project_lidar_to_image(self, lidar_ranges: np.ndarray) -> np.ndarray:
        """
        Project LiDAR points onto camera image
        :param lidar_ranges: LiDAR range measurements
        :return: Image with projected LiDAR points
        """
        # Create a dummy image
        img = np.zeros((self.camera.height, self.camera.width, 3), dtype=np.uint8)

        # Convert LiDAR ranges to 3D points
        lidar_points = self.lidar.ranges_to_cartesian(lidar_ranges)

        # Transform points to camera frame
        # For simplicity, assume LiDAR and camera are at same position
        camera_points = lidar_points

        # Project to 2D image
        if len(camera_points) > 0:
            points_2d = self.camera.project_3d_to_2d(camera_points)

            # Draw points on image
            for point in points_2d:
                u, v = int(point[0]), int(point[1])
                if 0 <= u < self.camera.width and 0 <= v < self.camera.height:
                    cv2.circle(img, (u, v), 2, (0, 255, 0), -1)

        return img

# Example usage
if __name__ == "__main__":
    # Create sensor fusion system
    fusion = SensorFusion()

    # Simulate some environment points (obstacles)
    np.random.seed(42)
    env_points = np.random.rand(50, 3) * 5  # Random points in 5x5x5 space
    env_points[:, 2] = 0  # Make them 2D for simplicity

    # Get LiDAR scan
    lidar_scan = fusion.lidar.scan(env_points)
    print(f"LiDAR scan completed with {len(lidar_scan)} beams")

    # Create occupancy grid
    occupancy_grid = fusion.get_occupancy_grid(lidar_scan)
    print(f"Occupancy grid created: {occupancy_grid.shape}")

    # Project LiDAR to camera image
    projected_img = fusion.project_lidar_to_image(lidar_scan)
    print(f"Projection completed: {projected_img.shape}")
```

## Communication Protocols

### CAN Bus Integration

```python
#!/usr/bin/env python3

import struct
import threading
import time
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass

@dataclass
class CANMessage:
    """CAN message structure"""
    id: int
    data: bytes
    timestamp: float
    extended: bool = True

class CANBus:
    """Simulated CAN bus implementation for robot communication"""

    def __init__(self, baud_rate: int = 500000):
        self.baud_rate = baud_rate
        self.bus_lock = threading.Lock()
        self.message_queue = []
        self.listeners: Dict[int, List[Callable]] = {}
        self.is_running = True

        # Start CAN bus simulation thread
        self.bus_thread = threading.Thread(target=self._bus_worker)
        self.bus_thread.start()

    def send_message(self, message: CANMessage):
        """Send a message on the CAN bus"""
        with self.bus_lock:
            self.message_queue.append(message)

    def add_listener(self, msg_id: int, callback: Callable[[CANMessage], None]):
        """Add a listener for specific message ID"""
        if msg_id not in self.listeners:
            self.listeners[msg_id] = []
        self.listeners[msg_id].append(callback)

    def _bus_worker(self):
        """Background thread to process messages"""
        while self.is_running:
            with self.bus_lock:
                messages_to_process = self.message_queue.copy()
                self.message_queue.clear()

            for msg in messages_to_process:
                # Notify listeners for this message ID
                if msg.id in self.listeners:
                    for callback in self.listeners[msg.id]:
                        try:
                            callback(msg)
                        except Exception as e:
                            print(f"Error in CAN listener: {e}")

            time.sleep(0.001)  # Simulate bus timing

    def stop(self):
        """Stop the CAN bus"""
        self.is_running = False
        if self.bus_thread.is_alive():
            self.bus_thread.join()

class MotorController:
    """Motor controller interface via CAN bus"""

    def __init__(self, can_bus: CANBus, node_id: int):
        self.can_bus = can_bus
        self.node_id = node_id
        self.position = 0.0
        self.velocity = 0.0
        self.current = 0.0

        # Register for motor feedback messages
        feedback_id = self.node_id | 0x100  # Feedback message ID
        self.can_bus.add_listener(feedback_id, self._handle_feedback)

    def set_velocity(self, velocity: float):
        """Set motor velocity"""
        # Create command message (velocity control)
        cmd_id = self.node_id | 0x200  # Command message ID
        data = struct.pack('<f', velocity)  # Pack velocity as float
        msg = CANMessage(id=cmd_id, data=data, timestamp=time.time())
        self.can_bus.send_message(msg)

    def set_position(self, position: float):
        """Set motor position"""
        # Create command message (position control)
        cmd_id = self.node_id | 0x201  # Position command
        data = struct.pack('<f', position)  # Pack position as float
        msg = CANMessage(id=cmd_id, data=data, timestamp=time.time())
        self.can_bus.send_message(msg)

    def _handle_feedback(self, msg: CANMessage):
        """Handle motor feedback message"""
        if len(msg.data) >= 12:  # Expect position, velocity, current
            self.position, self.velocity, self.current = struct.unpack('<fff', msg.data[:12])

class SensorNode:
    """Sensor node that publishes data via CAN bus"""

    def __init__(self, can_bus: CANBus, node_id: int, sensor_type: str):
        self.can_bus = can_bus
        self.node_id = node_id
        self.sensor_type = sensor_type
        self.data = None

        # Start sensor reading thread
        self.is_reading = True
        self.reading_thread = threading.Thread(target=self._read_sensor)
        self.reading_thread.start()

    def _read_sensor(self):
        """Simulate sensor reading"""
        while self.is_reading:
            # Simulate sensor reading
            if self.sensor_type == 'IMU':
                # Simulate IMU data: [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
                data = [0.0, 0.0, 9.81, 0.0, 0.0, 0.0]  # Static IMU reading
                packed_data = struct.pack('<ffffff', *data)
            elif self.sensor_type == 'ENCODER':
                # Simulate encoder data: [position, velocity]
                import random
                data = [random.uniform(-10, 10), random.uniform(-1, 1)]
                packed_data = struct.pack('<ff', *data)
            else:
                packed_data = b'\x00' * 8  # Default 8-byte payload

            # Send sensor data
            data_id = self.node_id | 0x300  # Sensor data ID
            msg = CANMessage(id=data_id, data=packed_data, timestamp=time.time())
            self.can_bus.send_message(msg)

            time.sleep(0.1)  # 10 Hz sensor rate

    def stop(self):
        """Stop sensor reading"""
        self.is_reading = False
        if self.reading_thread.is_alive():
            self.reading_thread.join()

class RobotController:
    """Main robot controller that coordinates all subsystems"""

    def __init__(self):
        # Initialize CAN bus
        self.can_bus = CANBus()

        # Initialize motor controllers (assuming 4 motors for a quad robot)
        self.motors = {
            'front_left': MotorController(self.can_bus, 0x01),
            'front_right': MotorController(self.can_bus, 0x02),
            'rear_left': MotorController(self.can_bus, 0x03),
            'rear_right': MotorController(self.can_bus, 0x04)
        }

        # Initialize sensor nodes
        self.imu_node = SensorNode(self.can_bus, 0x10, 'IMU')
        self.encoder_node = SensorNode(self.can_bus, 0x11, 'ENCODER')

        # Robot state
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.position = [0.0, 0.0]
        self.orientation = 0.0

    def set_cmd_vel(self, linear: float, angular: float):
        """Set robot linear and angular velocity"""
        self.linear_velocity = linear
        self.angular_velocity = angular

        # Convert to wheel velocities for differential drive
        wheel_separation = 0.3  # 30cm wheel separation
        wheel_radius = 0.05     # 5cm wheel radius

        # Differential drive kinematics
        v_left = (linear - angular * wheel_separation / 2) / wheel_radius
        v_right = (linear + angular * wheel_separation / 2) / wheel_radius

        # Set motor velocities
        self.motors['front_left'].set_velocity(v_left)
        self.motors['rear_left'].set_velocity(v_left)
        self.motors['front_right'].set_velocity(v_right)
        self.motors['rear_right'].set_velocity(v_right)

    def get_robot_state(self) -> Dict:
        """Get current robot state"""
        return {
            'position': self.position,
            'orientation': self.orientation,
            'linear_velocity': self.linear_velocity,
            'angular_velocity': self.angular_velocity,
            'motors': {name: {'position': m.position, 'velocity': m.velocity, 'current': m.current}
                      for name, m in self.motors.items()}
        }

    def stop(self):
        """Stop the robot controller"""
        self.set_cmd_vel(0.0, 0.0)
        self.can_bus.stop()
        self.imu_node.stop()
        self.encoder_node.stop()

# Example usage
if __name__ == "__main__":
    # Create robot controller
    robot = RobotController()

    # Command the robot to move forward and turn
    robot.set_cmd_vel(linear=0.5, angular=0.2)  # 0.5 m/s forward, 0.2 rad/s turn

    # Wait and get state
    time.sleep(2)
    state = robot.get_robot_state()
    print(f"Robot state: {state}")

    # Stop the robot
    robot.set_cmd_vel(0.0, 0.0)
    print("Robot stopped")

    # Clean up
    robot.stop()
```

## Practical Exercises

### Exercise 1: Build a Mobile Robot Platform

**Objective**: Design and simulate a complete mobile robot platform with appropriate locomotion, sensing, and control systems.

**Steps**:
1. Choose an appropriate locomotion system for your application
2. Select sensors based on the robot's intended tasks
3. Design the mechanical structure to support all components
4. Implement a basic control system to coordinate subsystems
5. Test the platform in simulation with various scenarios

**Expected Outcome**: A well-designed mobile robot platform with integrated hardware components and communication protocols.

### Exercise 2: Sensor Fusion Implementation

**Objective**: Implement a sensor fusion system that combines data from multiple sensors to improve robot perception.

**Steps**:
1. Integrate IMU, camera, and LiDAR data
2. Implement a Kalman filter or complementary filter
3. Test the system with simulated sensor data
4. Evaluate the improvement in state estimation
5. Analyze the impact of different sensor configurations

**Expected Outcome**: A functional sensor fusion system that provides more accurate state estimation than individual sensors.

### Exercise 3: Hardware-Software Integration

**Objective**: Create a complete hardware integration framework that connects various robot subsystems.

**Steps**:
1. Implement communication protocols (CAN, I2C, SPI)
2. Create device drivers for sensors and actuators
3. Design a modular software architecture
4. Implement error handling and fault tolerance
5. Test the integration with real or simulated hardware

**Expected Outcome**: A robust hardware integration framework that can reliably coordinate multiple robot subsystems.

## Chapter Summary

This chapter covered the essential aspects of mobile robot platforms and hardware integration:

1. **Locomotion Systems**: Different types of robot locomotion including wheeled, legged, and tracked systems, with detailed implementations of kinematic models.

2. **Sensor Integration**: Techniques for integrating various sensors including IMUs, cameras, and LiDAR, with sensor fusion approaches to combine multiple data sources.

3. **Communication Protocols**: Implementation of communication systems including CAN bus for coordinating distributed robot subsystems.

4. **Hardware Architecture**: Design principles for modular, scalable robot hardware platforms that can accommodate various sensors and actuators.

5. **System Integration**: Approaches for connecting hardware components with software systems to create cohesive robotic platforms.

The key to successful mobile robot development lies in the careful integration of mechanical design, sensor systems, communication protocols, and control algorithms. A well-integrated system provides reliable operation, robust sensing, and effective task execution.

## Further Reading

1. "Introduction to Autonomous Mobile Robots" by Siegwart et al. - Comprehensive coverage of mobile robot design
2. "Robotics, Vision and Control" by Corke - Practical robotics with MATLAB examples
3. "Probabilistic Robotics" by Thrun et al. - Sensor fusion and state estimation
4. "Handbook of Robotics" by Siciliano and Khatib - Comprehensive reference on robotics
5. "Embedded Robotics" by Tokhi et al. - Focus on embedded systems for robotics applications

## Assessment Questions

1. Compare different locomotion systems (wheeled, legged, tracked) in terms of efficiency, mobility, and complexity.

2. Derive the kinematic equations for a differential drive robot and explain how to control it.

3. Implement a sensor fusion algorithm that combines IMU and encoder data for position estimation.

4. Design a CAN bus communication protocol for a multi-robot system with various sensors and actuators.

5. Explain the design considerations for building a modular robot hardware platform.

6. Analyze the trade-offs between different wheel configurations (differential, Ackermann, omnidirectional).

7. Describe the implementation of a fault-tolerant sensor integration system.

8. Design a communication architecture for coordinating multiple robot subsystems.

9. Evaluate the performance of different sensor fusion approaches for mobile robot localization.

10. Discuss the challenges and solutions in integrating diverse hardware components in a robotic system.

