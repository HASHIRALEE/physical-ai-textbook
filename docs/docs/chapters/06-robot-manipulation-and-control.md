---
sidebar_position: 6
title: "Chapter 6: Robot Manipulation and Control"
---

# Chapter 6: Robot Manipulation and Control

## Learning Objectives

By the end of this chapter, students will be able to:
- Understand the fundamentals of robot manipulation including kinematics and dynamics
- Implement forward and inverse kinematics solutions for robotic arms
- Design and implement control strategies for precise manipulation tasks
- Apply trajectory planning algorithms for smooth and efficient motion
- Integrate force and tactile sensing for compliant manipulation
- Evaluate different control architectures for manipulation systems
- Implement grasping and manipulation algorithms in ROS2
- Assess the challenges and solutions in dexterous robotic manipulation

## Theoretical Foundations

### Kinematics of Robotic Manipulators

Robot manipulation begins with understanding the mathematical relationships between joint space and Cartesian space. Kinematics describes the motion of mechanical systems without considering the forces that cause the motion. For robotic manipulators, we distinguish between forward kinematics and inverse kinematics.

Forward kinematics involves calculating the end-effector position and orientation given the joint angles. This is accomplished through the use of transformation matrices, typically using the Denavit-Hartenberg (DH) convention to systematically define coordinate frames on each link of the manipulator.

The DH parameters consist of four values for each joint: link length (a), link twist (α), link offset (d), and joint angle (θ). These parameters define the transformation from one link frame to the next, and the complete transformation from the base to the end-effector is obtained by multiplying all individual transformations.

Inverse kinematics, conversely, involves determining the joint angles required to achieve a desired end-effector position and orientation. This is generally more complex than forward kinematics and may have multiple solutions, a unique solution, or no solution depending on the manipulator configuration and desired pose.

### Dynamics of Manipulation

The dynamics of robotic manipulators describe the relationship between forces/torques applied to the joints and the resulting motion. The equations of motion for a manipulator are typically expressed using the Lagrangian formulation:

M(q)q̈ + C(q, q̇)q̇ + G(q) = τ

Where:
- M(q) is the mass matrix
- C(q, q̇) represents Coriolis and centrifugal forces
- G(q) represents gravitational forces
- τ represents the joint torques
- q, q̇, q̈ are the joint positions, velocities, and accelerations

Understanding these dynamics is crucial for implementing model-based control strategies that can achieve precise and stable manipulation.

### Control Theory for Manipulation

Control strategies for robotic manipulation can be broadly categorized into position control, force control, and hybrid position/force control. Position control aims to track desired joint or Cartesian trajectories, while force control regulates contact forces during manipulation tasks. Hybrid control combines both approaches, which is essential for tasks involving interaction with the environment.

Impedance control is a popular approach that defines the relationship between position and force deviations, allowing the robot to behave like a virtual spring-damper system. This approach provides compliance that is essential for safe and robust manipulation.

## Forward and Inverse Kinematics

### Forward Kinematics Implementation

Let's implement forward kinematics for a 6-DOF robotic arm using the DH parameters:

```python
#!/usr/bin/env python3

import numpy as np
import math
from scipy.spatial.transform import Rotation as R

class ForwardKinematics:
    def __init__(self):
        # DH parameters for a 6-DOF manipulator (example: modified PUMA 560)
        self.dh_params = [
            {'a': 0, 'alpha': -np.pi/2, 'd': 0.4, 'theta': 0},      # Joint 1
            {'a': 0.4, 'alpha': 0, 'd': 0, 'theta': 0},             # Joint 2
            {'a': 0, 'alpha': np.pi/2, 'd': 0.14, 'theta': 0},      # Joint 3
            {'a': 0, 'alpha': -np.pi/2, 'd': 0.4, 'theta': 0},      # Joint 4
            {'a': 0, 'alpha': np.pi/2, 'd': 0, 'theta': 0},         # Joint 5
            {'a': 0, 'alpha': 0, 'd': 0.12, 'theta': 0}             # Joint 6
        ]

    def dh_transform(self, a, alpha, d, theta):
        """Calculate the DH transformation matrix"""
        return np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]
        ])

    def calculate_forward_kinematics(self, joint_angles):
        """Calculate forward kinematics for given joint angles"""
        if len(joint_angles) != len(self.dh_params):
            raise ValueError("Number of joint angles must match number of joints")

        # Update DH parameters with current joint angles
        for i, param in enumerate(self.dh_params):
            param['theta'] = joint_angles[i]

        # Calculate transformation matrices for each joint
        T_total = np.eye(4)
        transforms = []

        for param in self.dh_params:
            T = self.dh_transform(param['a'], param['alpha'], param['d'], param['theta'])
            T_total = np.dot(T_total, T)
            transforms.append(T_total.copy())

        return T_total, transforms

    def get_end_effector_pose(self, joint_angles):
        """Get end-effector position and orientation"""
        T, _ = self.calculate_forward_kinematics(joint_angles)

        # Extract position
        position = T[:3, 3]

        # Extract orientation (rotation matrix)
        rotation_matrix = T[:3, :3]

        # Convert to Euler angles for easier interpretation
        rotation = R.from_matrix(rotation_matrix)
        euler_angles = rotation.as_euler('xyz', degrees=True)

        return position, rotation_matrix, euler_angles

# Example usage
if __name__ == "__main__":
    fk = ForwardKinematics()

    # Example joint angles (in radians)
    joint_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    position, rotation_matrix, euler_angles = fk.get_end_effector_pose(joint_angles)

    print(f"End-effector position: {position}")
    print(f"End-effector orientation (Euler angles): {euler_angles}")
```

### Inverse Kinematics Implementation

Now let's implement an inverse kinematics solver using the Jacobian transpose method:

```python
#!/usr/bin/env python3

import numpy as np
from scipy.spatial.transform import Rotation as R

class InverseKinematics:
    def __init__(self):
        # Same DH parameters as forward kinematics
        self.dh_params = [
            {'a': 0, 'alpha': -np.pi/2, 'd': 0.4, 'theta': 0},
            {'a': 0.4, 'alpha': 0, 'd': 0, 'theta': 0},
            {'a': 0, 'alpha': np.pi/2, 'd': 0.14, 'theta': 0},
            {'a': 0, 'alpha': -np.pi/2, 'd': 0.4, 'theta': 0},
            {'a': 0, 'alpha': np.pi/2, 'd': 0, 'theta': 0},
            {'a': 0, 'alpha': 0, 'd': 0.12, 'theta': 0}
        ]

        self.fk = ForwardKinematics()

    def calculate_jacobian(self, joint_angles):
        """Calculate the geometric Jacobian matrix"""
        # Get current end-effector pose
        current_pose, _, _ = self.fk.get_end_effector_pose(joint_angles)
        current_position = current_pose[:3]

        # Initialize Jacobian (6x6 for position and orientation)
        jacobian = np.zeros((6, len(joint_angles)))

        # Calculate Jacobian columns
        for i in range(len(joint_angles)):
            # Update DH parameters with current joint angles
            for j, param in enumerate(self.dh_params):
                if j == i:
                    param['theta'] = joint_angles[j]
                else:
                    param['theta'] = 0  # Only this joint is active for Jacobian calculation

            # Calculate transformation up to joint i
            T_temp = np.eye(4)
            for param in self.dh_params[:i+1]:
                T = self.dh_transform(param['a'], param['alpha'], param['d'], param['theta'])
                T_temp = np.dot(T_temp, T)

            # Get joint position
            joint_pos = T_temp[:3, 3]

            # Get z-axis of joint frame
            z_axis = T_temp[:3, 2]

            # Position part of Jacobian
            jacobian[:3, i] = np.cross(z_axis, (current_position - joint_pos))

            # Orientation part of Jacobian
            jacobian[3:, i] = z_axis

        return jacobian

    def inverse_kinematics_jacobian(self, target_position, target_orientation,
                                   initial_joints, max_iterations=1000, tolerance=1e-4):
        """Solve inverse kinematics using Jacobian transpose method"""
        current_joints = np.array(initial_joints)

        for iteration in range(max_iterations):
            # Calculate current end-effector pose
            current_pos, current_rot, _ = self.fk.get_end_effector_pose(current_joints)

            # Calculate position error
            pos_error = target_position - current_pos

            # Calculate orientation error (using rotation matrix difference)
            target_rot_matrix = R.from_euler('xyz', target_orientation, degrees=True).as_matrix()
            rot_error_matrix = np.dot(target_rot_matrix, current_rot.T) - np.eye(3)

            # Convert rotation error to axis-angle representation
            rot_error = self.rotation_matrix_to_axis_angle(rot_error_matrix)

            # Combine position and orientation errors
            error = np.concatenate([pos_error, rot_error])

            # Check if error is within tolerance
            if np.linalg.norm(error) < tolerance:
                print(f"Converged after {iteration} iterations")
                return current_joints

            # Calculate Jacobian
            jacobian = self.calculate_jacobian(current_joints)

            # Calculate joint updates using Jacobian transpose
            joint_updates = np.dot(jacobian.T, error) * 0.01  # Small step size

            # Update joint angles
            current_joints += joint_updates

            # Apply joint limits if needed
            # current_joints = np.clip(current_joints, joint_limits_min, joint_limits_max)

        print(f"Did not converge after {max_iterations} iterations")
        return current_joints

    def rotation_matrix_to_axis_angle(self, rot_matrix):
        """Convert rotation matrix to axis-angle representation"""
        # Get rotation vector from rotation matrix
        rotation = R.from_matrix(rot_matrix)
        axis_angle = rotation.as_rotvec()
        return axis_angle

    def dh_transform(self, a, alpha, d, theta):
        """Calculate the DH transformation matrix"""
        return np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]
        ])

# Example usage
if __name__ == "__main__":
    ik = InverseKinematics()

    # Target pose
    target_pos = np.array([0.5, 0.2, 0.3])
    target_orientation = np.array([0, 0, 0])  # Euler angles in degrees

    # Initial joint configuration
    initial_joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Solve inverse kinematics
    solution = ik.inverse_kinematics_jacobian(target_pos, target_orientation, initial_joints)

    print(f"Joint angles solution: {solution}")

    # Verify solution
    fk = ForwardKinematics()
    final_pos, final_rot, final_euler = fk.get_end_effector_pose(solution)
    print(f"Final position: {final_pos}")
    print(f"Target position: {target_pos}")
    print(f"Position error: {np.linalg.norm(target_pos - final_pos)}")
```

## Control Strategies for Manipulation

### PID Control for Joint Position Control

```python
#!/usr/bin/env python3

import numpy as np
import time

class JointPIDController:
    def __init__(self, kp=10.0, ki=0.1, kd=0.01, joint_limits=None):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain

        self.joint_limits = joint_limits  # [min, max] for each joint

        # Error tracking
        self.prev_error = 0
        self.integral_error = 0

        # Time tracking
        self.prev_time = None

    def update(self, current_position, target_position, dt=None):
        """Update PID controller and return control output"""
        current_time = time.time()

        if dt is None:
            if self.prev_time is not None:
                dt = current_time - self.prev_time
            else:
                dt = 0.01  # Default time step
            self.prev_time = current_time
        else:
            self.prev_time = current_time

        # Calculate error
        error = target_position - current_position

        # Calculate integral (with anti-windup)
        self.integral_error += error * dt
        # Apply integral windup protection
        max_integral = 10.0
        self.integral_error = np.clip(self.integral_error, -max_integral, max_integral)

        # Calculate derivative
        if dt > 0:
            derivative = (error - self.prev_error) / dt
        else:
            derivative = 0

        # Calculate PID output
        output = (self.kp * error) + (self.ki * self.integral_error) + (self.kd * derivative)

        # Store current error for next iteration
        self.prev_error = error

        # Apply joint limits if specified
        if self.joint_limits is not None:
            output = np.clip(output, self.joint_limits[0], self.joint_limits[1])

        return output

class ManipulatorController:
    def __init__(self, num_joints=6):
        self.num_joints = num_joints
        self.controllers = [JointPIDController() for _ in range(num_joints)]

        # Current joint states
        self.current_positions = np.zeros(num_joints)
        self.current_velocities = np.zeros(num_joints)
        self.target_positions = np.zeros(num_joints)

        # Trajectory tracking
        self.trajectory = []
        self.trajectory_index = 0

    def set_target_positions(self, target_positions):
        """Set target joint positions"""
        if len(target_positions) != self.num_joints:
            raise ValueError("Target positions must match number of joints")
        self.target_positions = np.array(target_positions)

    def update_control(self, current_positions, dt=0.01):
        """Update control for all joints"""
        if len(current_positions) != self.num_joints:
            raise ValueError("Current positions must match number of joints")

        self.current_positions = np.array(current_positions)

        # Calculate control outputs for each joint
        control_outputs = []
        for i in range(self.num_joints):
            output = self.controllers[i].update(
                self.current_positions[i],
                self.target_positions[i],
                dt
            )
            control_outputs.append(output)

        return np.array(control_outputs)

    def move_to_position(self, target_positions, duration=5.0, steps=100):
        """Plan and execute smooth movement to target position"""
        start_positions = self.current_positions.copy()

        # Generate trajectory
        trajectory = []
        for i in range(steps + 1):
            t = i / steps  # Normalized time (0 to 1)
            # Use cubic interpolation for smooth motion
            interp_factor = 3*t**2 - 2*t**3
            current_pos = start_positions + interp_factor * (target_positions - start_positions)
            trajectory.append(current_pos)

        # Execute trajectory
        dt = duration / steps
        for pos in trajectory:
            control_output = self.update_control(self.current_positions, dt)
            # In a real system, you would send control_output to the robot
            # For simulation, we'll just update current positions
            self.current_positions = pos
            time.sleep(dt)  # Simulate time delay

# Example usage
if __name__ == "__main__":
    controller = ManipulatorController(6)

    # Move from initial position to target position
    initial_pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    target_pos = [0.5, 0.3, -0.2, 0.1, 0.4, -0.1]

    controller.current_positions = np.array(initial_pos)
    controller.set_target_positions(target_pos)

    print("Moving to target position...")
    controller.move_to_position(target_pos, duration=3.0, steps=100)
    print("Movement completed!")
```

### Impedance Control Implementation

```python
#!/usr/bin/env python3

import numpy as np

class ImpedanceController:
    def __init__(self, mass=1.0, damping=10.0, stiffness=100.0):
        """
        Initialize impedance controller with mass, damping, and stiffness parameters
        """
        self.mass = mass
        self.damping = damping
        self.stiffness = stiffness

        # Desired equilibrium position
        self.desired_position = np.zeros(3)
        self.desired_velocity = np.zeros(3)
        self.desired_acceleration = np.zeros(3)

        # Current state
        self.current_position = np.zeros(3)
        self.current_velocity = np.zeros(3)

        # Previous state for numerical differentiation
        self.prev_position = np.zeros(3)
        self.prev_time = None

    def update(self, current_position, external_force=None, dt=0.01):
        """
        Update impedance controller and return desired acceleration
        """
        if external_force is None:
            external_force = np.zeros(3)

        # Calculate current velocity using numerical differentiation
        current_time = time.time()
        if self.prev_time is not None:
            dt = current_time - self.prev_time
            if dt > 0:
                self.current_velocity = (current_position - self.prev_position) / dt

        self.current_position = current_position
        self.prev_position = current_position
        self.prev_time = current_time

        # Calculate position and velocity errors
        pos_error = self.desired_position - self.current_position
        vel_error = self.desired_velocity - self.current_velocity

        # Calculate impedance force
        impedance_force = (
            self.stiffness * pos_error +
            self.damping * vel_error +
            self.mass * self.desired_acceleration
        )

        # Total force (impedance + external)
        total_force = impedance_force + external_force

        # Calculate resulting acceleration (F = ma -> a = F/m)
        acceleration = total_force / self.mass

        return acceleration

    def set_desired_trajectory(self, position, velocity=None, acceleration=None):
        """Set desired trajectory parameters"""
        self.desired_position = np.array(position)
        if velocity is not None:
            self.desired_velocity = np.array(velocity)
        if acceleration is not None:
            self.desired_acceleration = np.array(acceleration)

class CartesianImpedanceController:
    def __init__(self, num_joints=6):
        self.num_joints = num_joints

        # Impedance controllers for each Cartesian DOF
        self.impedance_controllers = [
            ImpedanceController(mass=1.0, damping=10.0, stiffness=100.0)
            for _ in range(6)  # 3 for position, 3 for orientation
        ]

        # Jacobian for Cartesian to joint space conversion
        self.jacobian = np.eye(6, num_joints)  # Simplified identity for example

        # Robot kinematics
        self.fk = ForwardKinematics()
        self.ik = InverseKinematics()

    def update(self, current_joints, external_forces=None, dt=0.01):
        """
        Update Cartesian impedance controller
        """
        if external_forces is None:
            external_forces = np.zeros(6)

        # Calculate current Cartesian pose
        current_pos, current_rot, current_euler = self.fk.get_end_effector_pose(current_joints)
        current_cartesian = np.concatenate([current_pos, current_euler * np.pi / 180])  # Convert to radians

        # Update each impedance controller
        cartesian_accelerations = []
        for i in range(6):
            acc = self.impedance_controllers[i].update(
                current_cartesian[i],
                external_forces[i],
                dt
            )
            cartesian_accelerations.append(acc)

        cartesian_acceleration = np.array(cartesian_accelerations)

        # Convert Cartesian acceleration to joint space using Jacobian
        # τ = J^T * F (for force) or q̈ = J^(-1) * ẍ (for acceleration)
        try:
            # Use pseudo-inverse for non-square Jacobian
            jacobian_pinv = np.linalg.pinv(self.jacobian)
            joint_acceleration = np.dot(jacobian_pinv, cartesian_acceleration)
        except np.linalg.LinAlgError:
            # If Jacobian is singular, use damped least squares
            damping = 0.01
            jacobian_pinv = np.dot(
                self.jacobian.T,
                np.linalg.inv(np.dot(self.jacobian, self.jacobian.T) + damping**2 * np.eye(6))
            )
            joint_acceleration = np.dot(jacobian_pinv, cartesian_acceleration)

        return joint_acceleration

# Example usage
if __name__ == "__main__":
    import time

    # Initialize controller
    cartesian_controller = CartesianImpedanceController(6)

    # Set desired pose
    desired_pos = np.array([0.5, 0.2, 0.3])
    desired_orientation = np.array([0, 0, 0])  # In degrees
    desired_cartesian = np.concatenate([desired_pos, desired_orientation])

    for i in range(6):
        cartesian_controller.impedance_controllers[i].set_desired_trajectory(
            desired_cartesian[i]
        )

    # Simulate control loop
    current_joints = np.zeros(6)
    for step in range(100):
        joint_acc = cartesian_controller.update(current_joints, dt=0.01)

        # Simple integration to update joint positions
        current_joints += joint_acc * 0.01**2  # a*t^2

        print(f"Step {step}: Joints = {current_joints}")
        time.sleep(0.01)  # Simulate real-time control
```

## Trajectory Planning for Manipulation

### Joint Space Trajectory Planning

```python
#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

class JointSpaceTrajectoryPlanner:
    def __init__(self, num_joints=6):
        self.num_joints = num_joints

    def cubic_trajectory(self, start_pos, end_pos, duration, num_points=100):
        """
        Generate cubic polynomial trajectory
        q(t) = a0 + a1*t + a2*t^2 + a3*t^3
        """
        # Time vector
        t = np.linspace(0, duration, num_points)

        # Calculate polynomial coefficients
        # Boundary conditions: q(0)=start, q(T)=end, q'(0)=0, q'(T)=0
        a0 = start_pos
        a1 = 0  # Start velocity = 0
        a2 = (3/duration**2) * (end_pos - start_pos)
        a3 = (-2/duration**3) * (end_pos - start_pos)

        # Evaluate trajectory
        trajectory = a0 + a1*t + a2*t**2 + a3*t**3
        velocity = a1 + 2*a2*t + 3*a3*t**2
        acceleration = 2*a2 + 6*a3*t

        return {
            'time': t,
            'position': trajectory,
            'velocity': velocity,
            'acceleration': acceleration
        }

    def quintic_trajectory(self, start_pos, end_pos, start_vel=0, end_vel=0,
                          start_acc=0, end_acc=0, duration=1.0, num_points=100):
        """
        Generate quintic polynomial trajectory
        q(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        """
        # Time vector
        t = np.linspace(0, duration, num_points)

        # Calculate polynomial coefficients
        # Boundary conditions: position, velocity, and acceleration at start and end
        T = duration
        a0 = start_pos
        a1 = start_vel
        a2 = start_acc / 2

        # Solve for remaining coefficients using boundary conditions
        A = np.array([
            [T**3, T**4, T**5],
            [3*T**2, 4*T**3, 5*T**4],
            [6*T, 12*T**2, 20*T**3]
        ])

        b = np.array([
            end_pos - a0 - a1*T - a2*T**2,
            end_vel - a1 - 2*a2*T,
            end_acc - 2*a2
        ])

        a3, a4, a5 = np.linalg.solve(A, b)

        # Evaluate trajectory
        trajectory = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5
        velocity = a1 + 2*a2*t + 3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4
        acceleration = 2*a2 + 6*a3*t + 12*a4*t**2 + 20*a5*t**3

        return {
            'time': t,
            'position': trajectory,
            'velocity': velocity,
            'acceleration': acceleration
        }

    def multi_joint_trajectory(self, start_joints, end_joints, duration=5.0,
                              trajectory_type='quintic', num_points=500):
        """
        Generate trajectory for multiple joints
        """
        if len(start_joints) != self.num_joints or len(end_joints) != self.num_joints:
            raise ValueError("Start and end joint arrays must match number of joints")

        trajectories = []

        for i in range(self.num_joints):
            if trajectory_type == 'cubic':
                traj = self.cubic_trajectory(
                    start_joints[i], end_joints[i], duration, num_points
                )
            elif trajectory_type == 'quintic':
                traj = self.quintic_trajectory(
                    start_joints[i], end_joints[i],
                    duration=duration, num_points=num_points
                )
            else:
                raise ValueError("Trajectory type must be 'cubic' or 'quintic'")

            trajectories.append(traj)

        # Combine all joint trajectories
        combined_trajectory = {
            'time': trajectories[0]['time'],
            'position': np.array([traj['position'] for traj in trajectories]),
            'velocity': np.array([traj['velocity'] for traj in trajectories]),
            'acceleration': np.array([traj['acceleration'] for traj in trajectories])
        }

        return combined_trajectory

    def plot_trajectory(self, trajectory, joint_indices=None):
        """
        Plot trajectory for specified joints
        """
        if joint_indices is None:
            joint_indices = range(self.num_joints)

        fig, axes = plt.subplots(3, 1, figsize=(10, 12))

        for idx in joint_indices:
            axes[0].plot(trajectory['time'], trajectory['position'][idx],
                        label=f'Joint {idx+1}', linewidth=2)
            axes[1].plot(trajectory['time'], trajectory['velocity'][idx],
                        label=f'Joint {idx+1}', linewidth=2)
            axes[2].plot(trajectory['time'], trajectory['acceleration'][idx],
                        label=f'Joint {idx+1}', linewidth=2)

        axes[0].set_ylabel('Position (rad)')
        axes[0].set_title('Joint Position vs Time')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].set_ylabel('Velocity (rad/s)')
        axes[1].set_title('Joint Velocity vs Time')
        axes[1].legend()
        axes[1].grid(True)

        axes[2].set_ylabel('Acceleration (rad/s²)')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_title('Joint Acceleration vs Time')
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    planner = JointSpaceTrajectoryPlanner(6)

    # Define start and end joint positions
    start_joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    end_joints = np.array([1.0, 0.5, -0.5, 0.3, 0.2, -0.1])

    # Generate trajectory
    trajectory = planner.multi_joint_trajectory(
        start_joints, end_joints,
        duration=5.0,
        trajectory_type='quintic'
    )

    print("Trajectory generated successfully!")
    print(f"Trajectory duration: {trajectory['time'][-1]:.2f} seconds")
    print(f"Number of points: {len(trajectory['time'])}")

    # Plot trajectory for first 3 joints
    planner.plot_trajectory(trajectory, joint_indices=[0, 1, 2])
```

### Cartesian Space Trajectory Planning

```python
#!/usr/bin/env python3

import numpy as np
from scipy.spatial.transform import Rotation as R

class CartesianTrajectoryPlanner:
    def __init__(self, robot_kinematics):
        self.kinematics = robot_kinematics

    def linear_cartesian_trajectory(self, start_pose, end_pose, duration,
                                   num_points=100, orientation_blend=True):
        """
        Generate linear trajectory in Cartesian space
        pose format: [x, y, z, rx, ry, rz] where r is Euler angles in radians
        """
        # Extract positions and orientations
        start_pos = np.array(start_pose[:3])
        end_pos = np.array(end_pose[:3])

        start_rot = R.from_euler('xyz', start_pose[3:], degrees=False)
        end_rot = R.from_euler('xyz', end_pose[3:], degrees=False)

        # Time vector
        t = np.linspace(0, duration, num_points)
        alpha = t / duration  # Normalized time (0 to 1)

        # Generate position trajectory (linear interpolation)
        positions = start_pos + alpha[:, np.newaxis] * (end_pos - start_pos)

        # Generate orientation trajectory
        orientations = []
        if orientation_blend:
            # Use spherical linear interpolation (SLERP) for smooth rotation
            for a in alpha:
                interp_rot = R.slerp(start_rot, end_rot, a)
                orientations.append(interp_rot.as_euler('xyz'))
        else:
            # Linear interpolation of Euler angles (may not be smooth)
            euler_angles = start_rot.as_euler('xyz') + alpha[:, np.newaxis] * (
                end_rot.as_euler('xyz') - start_rot.as_euler('xyz')
            )
            orientations = euler_angles

        orientations = np.array(orientations)

        # Combine positions and orientations
        cartesian_trajectory = np.hstack([positions, orientations])

        # Calculate velocities and accelerations
        velocities = np.gradient(cartesian_trajectory, t[1] - t[0], axis=0)
        accelerations = np.gradient(velocities, t[1] - t[0], axis=0)

        return {
            'time': t,
            'position': cartesian_trajectory,
            'velocity': velocities,
            'acceleration': accelerations
        }

    def circular_cartesian_trajectory(self, center, radius, axis, angle_range,
                                    duration, num_points=100):
        """
        Generate circular trajectory in Cartesian space
        """
        # Time vector
        t = np.linspace(0, duration, num_points)
        alpha = t / duration  # Normalized time (0 to 1)

        # Calculate angles along the arc
        start_angle, end_angle = angle_range
        angles = start_angle + alpha * (end_angle - start_angle)

        # Generate circular path
        positions = []
        for angle in angles:
            # Create rotation matrix around the specified axis
            rot_matrix = R.from_rotvec(axis * angle).as_matrix()

            # Calculate position relative to center
            # For a circle in the plane perpendicular to the axis
            if np.allclose(axis, [1, 0, 0]):  # X-axis rotation
                local_pos = np.array([0, radius * np.cos(angle), radius * np.sin(angle)])
            elif np.allclose(axis, [0, 1, 0]):  # Y-axis rotation
                local_pos = np.array([radius * np.sin(angle), 0, radius * np.cos(angle)])
            else:  # Z-axis rotation
                local_pos = np.array([radius * np.cos(angle), radius * np.sin(angle), 0])

            # Transform to world coordinates and add to center
            world_pos = center + local_pos
            positions.append(world_pos)

        positions = np.array(positions)

        # For simplicity, keep orientation constant (you might want to implement orientation changes)
        orientations = np.zeros((num_points, 3))  # No orientation change

        # Combine positions and orientations
        cartesian_trajectory = np.hstack([positions, orientations])

        # Calculate velocities and accelerations
        velocities = np.gradient(cartesian_trajectory, t[1] - t[0], axis=0)
        accelerations = np.gradient(velocities, t[1] - t[0], axis=0)

        return {
            'time': t,
            'position': cartesian_trajectory,
            'velocity': velocities,
            'acceleration': accelerations
        }

    def convert_to_joint_trajectory(self, cartesian_trajectory, start_joints):
        """
        Convert Cartesian trajectory to joint space using inverse kinematics
        """
        ik = InverseKinematics()

        joint_trajectory = []
        current_joints = np.array(start_joints)

        for i, cart_pose in enumerate(cartesian_trajectory['position']):
            target_pos = cart_pose[:3]
            target_orientation = cart_pose[3:]  # In radians

            # Solve inverse kinematics
            # For smooth trajectory, use previous solution as initial guess
            solution = ik.inverse_kinematics_jacobian(
                target_pos,
                target_orientation * 180/np.pi,  # Convert to degrees for IK
                current_joints,
                max_iterations=100
            )

            joint_trajectory.append(solution)
            current_joints = solution  # Use current solution as next initial guess

        joint_trajectory = np.array(joint_trajectory)

        # Calculate joint velocities and accelerations
        dt = cartesian_trajectory['time'][1] - cartesian_trajectory['time'][0]
        joint_velocities = np.gradient(joint_trajectory, dt, axis=0)
        joint_accelerations = np.gradient(joint_velocities, dt, axis=0)

        return {
            'time': cartesian_trajectory['time'],
            'position': joint_trajectory,
            'velocity': joint_velocities,
            'acceleration': joint_accelerations
        }

# Example usage
if __name__ == "__main__":
    # Initialize planner with kinematics
    planner = CartesianTrajectoryPlanner(ForwardKinematics())

    # Define start and end poses [x, y, z, rx, ry, rz] in meters and radians
    start_pose = [0.5, 0.0, 0.3, 0.0, 0.0, 0.0]
    end_pose = [0.7, 0.2, 0.5, 0.2, 0.1, 0.05]

    # Generate Cartesian trajectory
    cart_trajectory = planner.linear_cartesian_trajectory(
        start_pose, end_pose, duration=5.0, num_points=200
    )

    print("Cartesian trajectory generated!")
    print(f"Start position: {start_pose[:3]}")
    print(f"End position: {end_pose[:3]}")
    print(f"Trajectory duration: {cart_trajectory['time'][-1]:.2f} seconds")

    # Convert to joint space (assuming we have start joint configuration)
    start_joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    joint_trajectory = planner.convert_to_joint_trajectory(cart_trajectory, start_joints)

    print(f"Joint trajectory generated with {len(joint_trajectory['time'])} points")
```

## Force and Tactile Sensing

### Force Control Implementation

```python
#!/usr/bin/env python3

import numpy as np

class ForceController:
    def __init__(self, kp=100.0, ki=10.0, kd=5.0):
        self.kp = kp  # Force proportional gain
        self.ki = ki  # Force integral gain
        self.kd = kd  # Force derivative gain

        # Force tracking
        self.desired_force = 0.0
        self.current_force = 0.0
        self.prev_force_error = 0.0
        self.integrated_force_error = 0.0

        # Time tracking
        self.prev_time = None

    def update(self, current_force, dt=None):
        """Update force controller and return position adjustment"""
        current_time = time.time()

        if dt is None:
            if self.prev_time is not None:
                dt = current_time - self.prev_time
            else:
                dt = 0.01  # Default time step
            self.prev_time = current_time
        else:
            self.prev_time = current_time

        # Calculate force error
        force_error = self.desired_force - current_force

        # Calculate integral (with anti-windup)
        self.integrated_force_error += force_error * dt
        max_integral = 50.0
        self.integrated_force_error = np.clip(
            self.integrated_force_error, -max_integral, max_integral
        )

        # Calculate derivative
        if dt > 0:
            force_derivative = (force_error - self.prev_force_error) / dt
        else:
            force_derivative = 0

        # Calculate force control output (position adjustment)
        position_adjustment = (
            self.kp * force_error +
            self.ki * self.integrated_force_error +
            self.kd * force_derivative
        )

        # Store current error for next iteration
        self.prev_force_error = force_error
        self.current_force = current_force

        return position_adjustment

class HybridPositionForceController:
    def __init__(self, num_dof=6):
        self.num_dof = num_dof

        # Separate controllers for each DOF
        self.position_controllers = [JointPIDController() for _ in range(num_dof)]
        self.force_controllers = [ForceController() for _ in range(num_dof)]

        # Selection matrix (1 for position control, 0 for force control)
        self.selection_matrix = np.ones(num_dof)  # Start with all position control

        # Current states
        self.current_positions = np.zeros(num_dof)
        self.current_forces = np.zeros(num_dof)
        self.target_positions = np.zeros(num_dof)
        self.target_forces = np.zeros(num_dof)

    def set_control_mode(self, dof_indices, mode='position'):
        """
        Set control mode for specific DOFs
        mode: 'position' or 'force'
        """
        if mode == 'position':
            self.selection_matrix[dof_indices] = 1
        elif mode == 'force':
            self.selection_matrix[dof_indices] = 0
        else:
            raise ValueError("Mode must be 'position' or 'force'")

    def set_target_positions(self, positions):
        """Set target positions for position-controlled DOFs"""
        if len(positions) != self.num_dof:
            raise ValueError("Positions array must match number of DOFs")
        self.target_positions = np.array(positions)

    def set_target_forces(self, forces):
        """Set target forces for force-controlled DOFs"""
        if len(forces) != self.num_dof:
            raise ValueError("Forces array must match number of DOFs")
        self.target_forces = np.array(forces)

    def update(self, current_positions, current_forces, dt=0.01):
        """Update hybrid controller"""
        if len(current_positions) != self.num_dof or len(current_forces) != self.num_dof:
            raise ValueError("Input arrays must match number of DOFs")

        self.current_positions = np.array(current_positions)
        self.current_forces = np.array(current_forces)

        # Calculate control outputs for each DOF
        control_outputs = []

        for i in range(self.num_dof):
            if self.selection_matrix[i] == 1:  # Position control
                output = self.position_controllers[i].update(
                    self.current_positions[i],
                    self.target_positions[i],
                    dt
                )
            else:  # Force control
                output = self.force_controllers[i].update(
                    self.current_forces[i],
                    dt
                )

            control_outputs.append(output)

        return np.array(control_outputs)

# Force/Torque sensor simulation
class ForceTorqueSensor:
    def __init__(self, noise_level=0.01):
        self.noise_level = noise_level
        self.bias = np.zeros(6)  # [Fx, Fy, Fz, Mx, My, Mz]

    def read(self, true_force_torque):
        """Simulate sensor reading with noise"""
        noise = np.random.normal(0, self.noise_level, 6)
        return true_force_torque + noise + self.bias

# Example usage
if __name__ == "__main__":
    # Initialize hybrid controller
    hybrid_controller = HybridPositionForceController(6)

    # Set first 3 DOFs to position control, last 3 to force control
    hybrid_controller.set_control_mode([0, 1, 2], 'position')
    hybrid_controller.set_control_mode([3, 4, 5], 'force')

    # Set targets
    hybrid_controller.set_target_positions([0.5, 0.2, 0.3, 0, 0, 0])
    hybrid_controller.set_target_forces([0, 0, 5.0, 0, 0, 0])  # 5N normal force

    # Simulate control loop
    current_positions = np.zeros(6)
    current_forces = np.zeros(6)

    force_sensor = ForceTorqueSensor(noise_level=0.01)

    for step in range(100):
        # Simulate sensor readings (in real system, these would come from actual sensors)
        noisy_forces = force_sensor.read(current_forces)

        # Update controller
        control_output = hybrid_controller.update(
            current_positions,
            noisy_forces,
            dt=0.01
        )

        # Update simulated robot state (in real system, send commands to robot)
        current_positions += control_output * 0.01  # Simple integration

        # Simulate force changes based on position (simplified contact model)
        # When z position is near contact, apply normal force
        if current_positions[2] < 0.1:  # Near contact surface
            current_forces[2] = 5.0 * (0.1 - current_positions[2])  # Spring-like contact
        else:
            current_forces[2] = 0  # No contact

        print(f"Step {step}: Pos={current_positions[:3]:.3f}, Forces={current_forces[2]:.3f}N")
        time.sleep(0.01)  # Simulate real-time control
```

## Practical Exercises

### Exercise 1: Implement a Simple Robotic Arm Controller

**Objective**: Create a complete control system for a 3-DOF robotic arm with position control and trajectory planning.

**Steps**:
1. Implement forward and inverse kinematics for a 3-DOF planar manipulator
2. Create a PID controller for each joint
3. Implement trajectory planning for smooth motion between waypoints
4. Simulate the arm's response to various commands
5. Test the system with different trajectory profiles

**Expected Outcome**: A functional simulation of a 3-DOF robotic arm that can follow planned trajectories while maintaining stable control.

### Exercise 2: Force-Controlled Grasping

**Objective**: Implement a force control system for robotic grasping that can adapt to different object properties.

**Steps**:
1. Set up a force-torque sensor simulation
2. Implement a hybrid position/force controller
3. Create a grasping controller that adjusts grip force based on object properties
4. Test with objects of different stiffness and fragility
5. Evaluate the controller's performance with various contact scenarios

**Expected Outcome**: A grasping system that can safely grasp objects of varying properties by controlling contact forces.

### Exercise 3: Cartesian Impedance Control

**Objective**: Implement Cartesian impedance control for safe human-robot interaction.

**Steps**:
1. Create an impedance controller in Cartesian space
2. Implement admittance control for human guidance
3. Add safety limits and constraints
4. Test the system's response to external forces
5. Evaluate compliance and safety characteristics

**Expected Outcome**: A compliant robotic system that can safely interact with humans while maintaining task performance.

## Chapter Summary

This chapter covered the fundamental concepts and implementation techniques for robot manipulation and control:

1. **Kinematics**: Understanding forward and inverse kinematics for robotic arms, with practical implementations using DH parameters and Jacobian matrices.

2. **Control Strategies**: Various control approaches including PID control, impedance control, and hybrid position/force control for different manipulation tasks.

3. **Trajectory Planning**: Methods for generating smooth, controlled motions in both joint and Cartesian space, with polynomial interpolation techniques.

4. **Force Sensing and Control**: Integration of force and tactile feedback for compliant manipulation and safe human-robot interaction.

5. **Practical Implementation**: Real-world considerations for implementing manipulation systems, including sensor integration, safety, and performance optimization.

The key to successful robot manipulation lies in the proper integration of kinematic models, control theory, and sensory feedback. Modern manipulation systems require sophisticated control architectures that can handle the complexity of real-world tasks while ensuring safety and efficiency.

## Further Reading

1. "Robotics: Modelling, Planning and Control" by Siciliano et al. - Comprehensive coverage of robotics fundamentals
2. "Handbook of Robotics" by Nof - Extensive reference on all aspects of robotics
3. "Robot Manipulator Control: Theory and Practice" by Lewis et al. - Detailed treatment of control theory
4. "Introduction to Robotics: Mechanics and Control" by Craig - Classic textbook on robot kinematics and dynamics
5. "Force and Motion Control of Robot Manipulators" by Colbaugh and Glass - Specialized focus on force control

## Assessment Questions

1. Derive the forward kinematics equations for a 6-DOF robotic arm using the Denavit-Hartenberg convention.

2. Compare and contrast the Jacobian transpose method, Jacobian pseudoinverse method, and closed-form solutions for inverse kinematics.

3. Explain the differences between position control, force control, and impedance control, providing examples of tasks where each approach is most appropriate.

4. Design a trajectory planning algorithm that generates smooth motion between two arbitrary poses while avoiding obstacles.

5. Describe the implementation of a hybrid position/force controller for a robotic assembly task.

6. Analyze the stability and performance of different PID tuning methods for robotic manipulator control.

7. Discuss the challenges and solutions involved in force-controlled manipulation of deformable objects.

8. Explain how to implement Cartesian impedance control and evaluate its effectiveness for human-robot collaboration.

9. Compare the advantages and disadvantages of joint-space vs. Cartesian-space control approaches.

10. Design a complete control system for a dexterous robotic hand with multiple fingers and joints.

