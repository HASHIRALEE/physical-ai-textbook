---
sidebar_position: 11
title: "Chapter 11: Control Systems for Robotics"
---

# Chapter 11: Control Systems for Robotics

## Learning Objectives

By the end of this chapter, students will be able to:
- Understand fundamental control theory concepts and their application to robotics
- Implement classical control algorithms like PID controllers for robotic systems
- Design advanced control strategies including adaptive and robust control
- Apply optimal control techniques such as LQR and MPC for robotic applications
- Implement feedback linearization and computed torque control methods
- Analyze stability and performance of robotic control systems
- Integrate control systems with robot perception and planning modules
- Evaluate different control architectures for various robotic applications

## Theoretical Foundations

### Introduction to Control Theory for Robotics

Control theory is fundamental to robotics, enabling robots to execute desired behaviors by manipulating their actuators based on sensor feedback. In robotics, control systems bridge the gap between high-level planning and low-level actuator commands, ensuring that robots can accurately track trajectories, maintain stability, and respond appropriately to environmental changes.

A robotic control system typically consists of:
- **Plant**: The physical robot system to be controlled
- **Controller**: The algorithm that computes control inputs
- **Sensors**: Devices that measure system state
- **Actuators**: Components that apply control inputs to the plant
- **Reference**: The desired behavior or trajectory

The mathematical foundation for robotic control systems is based on differential equations that describe the relationship between inputs, states, and outputs. For a robotic manipulator, the equation of motion is typically expressed as:

M(q)q̈ + C(q, q̇)q̇ + G(q) = τ

Where:
- M(q) is the mass matrix
- C(q, q̇) represents Coriolis and centrifugal forces
- G(q) represents gravitational forces
- τ represents joint torques
- q, q̇, q̈ are the joint positions, velocities, and accelerations

### Control System Classification

Control systems in robotics can be classified based on several criteria:

**Open-loop vs. Closed-loop**: Open-loop systems apply predetermined control inputs without feedback, while closed-loop systems use sensor feedback to adjust control actions.

**Linear vs. Nonlinear**: Linear systems follow the principle of superposition, while nonlinear systems (most robotic systems) do not.

**Time-invariant vs. Time-varying**: Time-invariant systems have parameters that do not change with time.

**Continuous vs. Discrete**: Continuous systems operate in continuous time, while discrete systems operate at discrete time intervals (common in digital implementations).

### Stability Analysis

Stability is a crucial property of control systems, ensuring that the system remains bounded and converges to desired states. For robotic systems, stability analysis often involves:

**Lyapunov Stability**: A system is Lyapunov stable if trajectories starting near an equilibrium point remain near it for all future time.

**Asymptotic Stability**: A system is asymptotically stable if it is Lyapunov stable and trajectories converge to the equilibrium point.

**BIBO Stability**: Bounded-input, bounded-output stability ensures that bounded inputs produce bounded outputs.

## Classical Control Methods

### PID Control Implementation

Proportional-Integral-Derivative (PID) control is the most widely used control strategy in robotics due to its simplicity and effectiveness:

```python
#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List
import time

class PIDController:
    def __init__(self, kp: float, ki: float, kd: float,
                 output_limits: tuple = (-1.0, 1.0),
                 setpoint: float = 0.0):
        """
        Initialize PID controller
        :param kp: Proportional gain
        :param ki: Integral gain
        :param kd: Derivative gain
        :param output_limits: Tuple of (min_output, max_output)
        :param setpoint: Initial setpoint
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.output_limits = output_limits

        # Internal variables
        self._last_error = 0.0
        self._integral = 0.0
        self._last_time = None
        self._last_derivative = 0.0

        # Anti-windup parameters
        self._windup_guard = 100.0

    def update(self, measurement: float, dt: float = None) -> float:
        """
        Update PID controller and return control output
        :param measurement: Current measured value
        :param dt: Time step (if None, calculated internally)
        :return: Control output
        """
        current_time = time.time()

        if dt is None:
            if self._last_time is not None:
                dt = current_time - self._last_time
            else:
                dt = 0.01  # Default time step
            self._last_time = current_time
        else:
            self._last_time = current_time

        # Calculate error
        error = self.setpoint - measurement

        # Calculate integral with anti-windup
        self._integral += error * dt
        # Clamp integral to prevent windup
        max_integral = self._windup_guard / self.ki if self.ki != 0 else self._windup_guard
        self._integral = np.clip(self._integral, -max_integral, max_integral)

        # Calculate derivative
        if dt > 0:
            derivative = (error - self._last_error) / dt
            # Use filtered derivative to reduce noise sensitivity
            alpha = 0.1  # Filter coefficient
            self._last_derivative = alpha * derivative + (1 - alpha) * self._last_derivative
        else:
            self._last_derivative = 0.0

        # Calculate PID output
        proportional = self.kp * error
        integral = self.ki * self._integral
        derivative_term = self.kd * self._last_derivative

        output = proportional + integral + derivative_term

        # Apply output limits
        output = np.clip(output, self.output_limits[0], self.output_limits[1])

        # Store current error for next iteration
        self._last_error = error

        return output

    def set_setpoint(self, setpoint: float):
        """Set new setpoint"""
        self.setpoint = setpoint

    def set_tunings(self, kp: float, ki: float, kd: float):
        """Set new PID tuning parameters"""
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def reset(self):
        """Reset internal variables"""
        self._last_error = 0.0
        self._integral = 0.0
        self._last_time = None
        self._last_derivative = 0.0

class MultiDOFPIDController:
    def __init__(self, num_joints: int, kp: Union[float, List[float]],
                 ki: Union[float, List[float]], kd: Union[float, List[float]]):
        """
        Multi-DOF PID controller for robotic systems
        :param num_joints: Number of joints to control
        :param kp: Proportional gains (single value or list)
        :param ki: Integral gains (single value or list)
        :param kd: Derivative gains (single value or list)
        """
        self.num_joints = num_joints

        # Handle both single values and lists for gains
        if isinstance(kp, (int, float)):
            kp = [kp] * num_joints
        if isinstance(ki, (int, float)):
            ki = [ki] * num_joints
        if isinstance(kd, (int, float)):
            kd = [kd] * num_joints

        # Create individual PID controllers for each joint
        self.controllers = []
        for i in range(num_joints):
            controller = PIDController(kp[i], ki[i], kd[i])
            self.controllers.append(controller)

    def update(self, current_positions: List[float], target_positions: List[float],
               dt: float = None) -> List[float]:
        """
        Update all PID controllers
        :param current_positions: Current joint positions
        :param target_positions: Target joint positions
        :param dt: Time step
        :return: Control outputs for each joint
        """
        if len(current_positions) != self.num_joints or len(target_positions) != self.num_joints:
            raise ValueError("Position lists must match number of joints")

        outputs = []
        for i in range(self.num_joints):
            self.controllers[i].setpoint = target_positions[i]
            output = self.controllers[i].update(current_positions[i], dt)
            outputs.append(output)

        return outputs

    def set_setpoints(self, setpoints: List[float]):
        """Set setpoints for all joints"""
        if len(setpoints) != self.num_joints:
            raise ValueError("Setpoints list must match number of joints")

        for i in range(self.num_joints):
            self.controllers[i].setpoint = setpoints[i]

# Example usage
if __name__ == "__main__":
    # Create a PID controller for a single joint
    pid = PIDController(kp=2.0, ki=0.1, kd=0.05, output_limits=(-10, 10))

    # Simulate control of a simple system
    current_value = 0.0
    target_value = 1.0
    dt = 0.01
    time_steps = 1000
    results = []

    for i in range(time_steps):
        pid.setpoint = target_value
        control_output = pid.update(current_value, dt)

        # Simulate system response (simple first-order system)
        current_value += (control_output - current_value * 0.1) * dt

        results.append((i * dt, current_value, control_output))

    print(f"PID control simulation completed. Final value: {current_value:.3f}")
```

### Advanced PID Techniques

```python
#!/usr/bin/env python3

import numpy as np
from typing import Tuple, Optional
import time

class AdvancedPIDController:
    def __init__(self, kp: float, ki: float, kd: float,
                 derivative_filter: float = 0.1,
                 setpoint_weighting: float = 1.0,
                 output_limits: Tuple[float, float] = (-1.0, 1.0)):
        """
        Advanced PID controller with additional features
        :param kp: Proportional gain
        :param ki: Integral gain
        :param kd: Derivative gain
        :param derivative_filter: Low-pass filter coefficient for derivative term
        :param setpoint_weighting: Weighting factor for setpoint in derivative term
        :param output_limits: Output limits (min, max)
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.derivative_filter = derivative_filter
        self.setpoint_weighting = setpoint_weighting
        self.output_limits = output_limits

        # Internal state
        self._last_error = 0.0
        self._integral = 0.0
        self._filtered_derivative = 0.0
        self._last_time = None
        self._last_measurement = 0.0

        # Anti-windup
        self._windup_guard = 100.0

    def update(self, measurement: float, setpoint: float, dt: float = None) -> float:
        """
        Update advanced PID controller
        :param measurement: Current measured value
        :param setpoint: Desired setpoint
        :param dt: Time step
        :return: Control output
        """
        current_time = time.time()

        if dt is None:
            if self._last_time is not None:
                dt = current_time - self._last_time
            else:
                dt = 0.01
            self._last_time = current_time
        else:
            self._last_time = current_time

        # Calculate error
        error = setpoint - measurement

        # Proportional term with setpoint weighting
        proportional = self.kp * (self.setpoint_weighting * (setpoint - self._last_measurement) - error)

        # Integral term with anti-windup
        self._integral += self.ki * error * dt

        # Anti-windup: limit integral based on output saturation
        output = proportional + self._integral + self._filtered_derivative
        if output > self.output_limits[1] and error > 0:
            self._integral -= self.ki * error * dt  # Don't integrate when saturated high
        elif output < self.output_limits[0] and error < 0:
            self._integral -= self.ki * error * dt  # Don't integrate when saturated low

        # Clamp integral to prevent excessive windup
        max_integral = self._windup_guard
        self._integral = np.clip(self._integral, -max_integral, max_integral)

        # Derivative term with low-pass filtering
        if dt > 0:
            raw_derivative = (measurement - self._last_measurement) / dt
            # Apply low-pass filter to derivative
            self._filtered_derivative = (
                self.derivative_filter * raw_derivative +
                (1 - self.derivative_filter) * self._filtered_derivative
            )
            # Apply derivative gain (negative because derivative of error is negative of measurement)
            derivative_term = -self.kd * self._filtered_derivative
        else:
            derivative_term = 0.0

        # Calculate total output
        output = proportional + self._integral + derivative_term

        # Apply output limits
        output = np.clip(output, self.output_limits[0], self.output_limits[1])

        # Update internal state
        self._last_error = error
        self._last_measurement = measurement

        return output

class AdaptivePIDController:
    def __init__(self, initial_kp: float = 1.0, initial_ki: float = 0.1, initial_kd: float = 0.01,
                 adaptation_rate: float = 0.01, max_gain: float = 10.0):
        """
        Adaptive PID controller that adjusts gains based on system performance
        :param initial_kp: Initial proportional gain
        :param initial_ki: Initial integral gain
        :param initial_kd: Initial derivative gain
        :param adaptation_rate: Rate of gain adaptation
        :param max_gain: Maximum allowed gain value
        """
        self.kp = initial_kp
        self.ki = initial_ki
        self.kd = initial_kd
        self.adaptation_rate = adaptation_rate
        self.max_gain = max_gain

        # Internal state for basic PID
        self._last_error = 0.0
        self._integral = 0.0
        self._last_time = None
        self._last_derivative = 0.0

        # Performance tracking for adaptation
        self._error_history = []
        self._max_history = 100  # Keep last 100 error samples

    def update(self, measurement: float, setpoint: float, dt: float = None) -> float:
        """
        Update adaptive PID controller
        :param measurement: Current measured value
        :param setpoint: Desired setpoint
        :param dt: Time step
        :return: Control output
        """
        current_time = time.time()

        if dt is None:
            if self._last_time is not None:
                dt = current_time - self._last_time
            else:
                dt = 0.01
            self._last_time = current_time
        else:
            self._last_time = current_time

        # Calculate error
        error = setpoint - measurement

        # Update error history for adaptation
        self._error_history.append(abs(error))
        if len(self._error_history) > self._max_history:
            self._error_history.pop(0)

        # Adapt gains based on performance
        self._adapt_gains()

        # Calculate PID terms
        self._integral += error * dt
        self._integral = np.clip(self._integral, -10, 10)  # Anti-windup

        if dt > 0:
            derivative = (error - self._last_error) / dt
            # Low-pass filter for derivative
            alpha = 0.1
            self._last_derivative = alpha * derivative + (1 - alpha) * self._last_derivative
        else:
            self._last_derivative = 0.0

        # Calculate output
        output = (self.kp * error +
                 self.ki * self._integral +
                 self.kd * self._last_derivative)

        # Clamp output
        output = np.clip(output, -10, 10)

        # Update internal state
        self._last_error = error

        return output

    def _adapt_gains(self):
        """Adapt PID gains based on system performance"""
        if len(self._error_history) < 10:
            return

        # Calculate performance metrics
        avg_error = np.mean(self._error_history)
        recent_error = np.mean(self._error_history[-10:])

        # Adjust gains based on error trends
        if avg_error > 0.1:  # High steady-state error
            self.kp = min(self.kp * 1.01, self.max_gain)
            self.ki = min(self.ki * 1.02, self.max_gain)
        elif recent_error < 0.01 and avg_error > 0.05:  # Oscillating
            self.kp = max(self.kp * 0.99, 0.1)
            self.kd = min(self.kd * 1.01, self.max_gain)

        # Apply adaptation rate limit
        self.kp = np.clip(self.kp, 0.1, self.max_gain)
        self.ki = np.clip(self.ki, 0.01, self.max_gain)
        self.kd = np.clip(self.kd, 0.001, self.max_gain)

# Example usage
if __name__ == "__main__":
    # Test advanced PID controller
    advanced_pid = AdvancedPIDController(
        kp=2.0, ki=0.1, kd=0.05,
        derivative_filter=0.2,
        setpoint_weighting=0.8
    )

    # Test adaptive PID controller
    adaptive_pid = AdaptivePIDController(
        initial_kp=1.0, initial_ki=0.05, initial_kd=0.01
    )

    print("Advanced PID and Adaptive PID controllers initialized")
```

## Modern Control Techniques

### Linear Quadratic Regulator (LQR)

```python
#!/usr/bin/env python3

import numpy as np
import scipy.linalg
from typing import Tuple, Optional
import matplotlib.pyplot as plt

class LQRController:
    def __init__(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray):
        """
        Linear Quadratic Regulator controller
        :param A: System dynamics matrix (state matrix)
        :param B: Input matrix
        :param Q: State cost matrix
        :param R: Input cost matrix
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R

        # Verify system dimensions
        n_states = A.shape[0]
        n_inputs = B.shape[1]

        if A.shape[0] != A.shape[1]:
            raise ValueError("A matrix must be square")
        if B.shape[0] != n_states:
            raise ValueError("B matrix rows must match A matrix dimensions")
        if Q.shape != A.shape:
            raise ValueError("Q matrix must have same dimensions as A")
        if R.shape[0] != R.shape[1] or R.shape[0] != n_inputs:
            raise ValueError("R matrix must be square with dimensions matching inputs")

        # Solve Riccati equation to find optimal gain matrix
        self.P = self._solve_riccati(A, B, Q, R)
        self.K = self._compute_gain(A, B, self.P, R)

    def _solve_riccati(self, A: np.ndarray, B: np.ndarray,
                      Q: np.ndarray, R: np.ndarray) -> np.ndarray:
        """
        Solve the continuous-time algebraic Riccati equation
        A^T * P + P * A - P * B * R^(-1) * B^T * P + Q = 0
        """
        P = scipy.linalg.solve_continuous_are(A, B, Q, R)
        return P

    def _compute_gain(self, A: np.ndarray, B: np.ndarray,
                     P: np.ndarray, R: np.ndarray) -> np.ndarray:
        """
        Compute the optimal feedback gain matrix K = R^(-1) * B^T * P
        """
        K = np.linalg.inv(R) @ B.T @ P
        return K

    def control(self, state: np.ndarray, reference: np.ndarray = None) -> np.ndarray:
        """
        Compute control input using LQR
        :param state: Current state vector
        :param reference: Reference state (if None, assumes zero reference)
        :return: Control input vector
        """
        if reference is None:
            reference = np.zeros_like(state)

        # Calculate error
        error = reference - state

        # Compute control input: u = K * (x_ref - x)
        control_input = self.K @ error

        return control_input

class DiscreteLQRController:
    def __init__(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray):
        """
        Discrete-time Linear Quadratic Regulator controller
        :param A: Discrete system dynamics matrix
        :param B: Discrete input matrix
        :param Q: State cost matrix
        :param R: Input cost matrix
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R

        # Verify system dimensions
        n_states = A.shape[0]
        n_inputs = B.shape[1]

        if A.shape[0] != A.shape[1]:
            raise ValueError("A matrix must be square")
        if B.shape[0] != n_states:
            raise ValueError("B matrix rows must match A matrix dimensions")

        # Solve discrete-time Riccati equation
        self.P = self._solve_discrete_riccati(A, B, Q, R)
        self.K = self._compute_discrete_gain(A, B, self.P, R)

    def _solve_discrete_riccati(self, A: np.ndarray, B: np.ndarray,
                               Q: np.ndarray, R: np.ndarray) -> np.ndarray:
        """
        Solve the discrete-time algebraic Riccati equation
        P = A^T * P * A - (A^T * P * B) * (R + B^T * P * B)^(-1) * (B^T * P * A) + Q
        """
        P = scipy.linalg.solve_discrete_are(A, B, Q, R)
        return P

    def _compute_discrete_gain(self, A: np.ndarray, B: np.ndarray,
                              P: np.ndarray, R: np.ndarray) -> np.ndarray:
        """
        Compute discrete-time feedback gain matrix
        K = (R + B^T * P * B)^(-1) * B^T * P * A
        """
        K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        return K

    def control(self, state: np.ndarray, reference: np.ndarray = None) -> np.ndarray:
        """
        Compute discrete control input using LQR
        :param state: Current state vector
        :param reference: Reference state
        :return: Control input vector
        """
        if reference is None:
            reference = np.zeros_like(state)

        error = reference - state
        control_input = self.K @ error

        return control_input

class MPCController:
    def __init__(self, A: np.ndarray, B: np.ndarray,
                 Q: np.ndarray, R: np.ndarray,
                 prediction_horizon: int = 10,
                 control_horizon: int = 5):
        """
        Model Predictive Control controller
        :param A: System dynamics matrix
        :param B: Input matrix
        :param Q: State cost matrix
        :param R: Input cost matrix
        :param prediction_horizon: Number of steps to predict ahead
        :param control_horizon: Number of steps to optimize control
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.prediction_horizon = prediction_horizon
        self.control_horizon = min(control_horizon, prediction_horizon)

        # System dimensions
        self.n_states = A.shape[0]
        self.n_inputs = B.shape[1]

        # Pre-compute prediction matrices for efficiency
        self._compute_prediction_matrices()

    def _compute_prediction_matrices(self):
        """Compute matrices for prediction over the horizon"""
        # State prediction matrix: X = Phi_x * x0 + Phi_u * U
        self.Phi_x = np.zeros((self.n_states * self.prediction_horizon, self.n_states))
        self.Phi_u = np.zeros((self.n_states * self.prediction_horizon,
                              self.n_inputs * self.control_horizon))

        # Build prediction matrices
        A_power = np.eye(self.n_states)
        for k in range(self.prediction_horizon):
            if k == 0:
                self.Phi_x[k*self.n_states:(k+1)*self.n_states, :] = A_power
            else:
                self.Phi_x[k*self.n_states:(k+1)*self.n_states, :] = A_power @ self.A

            # For each control step in the horizon
            for j in range(min(k+1, self.control_horizon)):
                if k >= j:
                    # Calculate A^(k-j) * B
                    A_to_power = np.linalg.matrix_power(self.A, k-j) if k-j > 0 else np.eye(self.n_states)
                    self.Phi_u[k*self.n_states:(k+1)*self.n_states,
                              j*self.n_inputs:(j+1)*self.n_inputs] = A_to_power @ self.B

            A_power = A_power @ self.A

    def control(self, state: np.ndarray, reference_trajectory: np.ndarray) -> np.ndarray:
        """
        Compute MPC control input
        :param state: Current state vector
        :param reference_trajectory: Reference trajectory [horizon x n_states]
        :return: Control input vector
        """
        # Construct quadratic programming problem
        # Minimize: (1/2) * U^T * H * U + f^T * U
        # Subject to constraints (simplified here)

        # Cost matrix for control inputs
        R_block = np.kron(np.eye(self.control_horizon), self.R)

        # State cost matrix
        Q_block = np.kron(np.eye(self.prediction_horizon), self.Q)

        # Combined H matrix for optimization
        H = self.Phi_u.T @ Q_block @ self.Phi_u + R_block

        # Linear term in cost function
        x_ref_vec = reference_trajectory.flatten()
        x0_vec = np.kron(np.ones(self.prediction_horizon), state)
        error_vec = x_ref_vec - self.Phi_x @ state

        f = self.Phi_u.T @ Q_block @ error_vec

        # Solve the quadratic program (simplified - in practice, use a QP solver)
        # For unconstrained case: U_opt = -H^(-1) * f
        try:
            U_opt = -np.linalg.solve(H, f)
            # Return only the first control input
            return U_opt[:self.n_inputs]
        except np.linalg.LinAlgError:
            # If H is singular, return zero control
            return np.zeros(self.n_inputs)

# Example: Create LQR controller for a simple 2nd order system
if __name__ == "__main__":
    # Example: Mass-spring-damper system: m*ẍ + c*ẋ + k*x = u
    # State: x1 = position, x2 = velocity
    # State-space: ẋ1 = x2, ẋ2 = -(c/m)*x2 - (k/m)*x1 + (1/m)*u

    m, c, k = 1.0, 0.5, 2.0  # mass, damping, stiffness

    A = np.array([
        [0, 1],
        [-k/m, -c/m]
    ])

    B = np.array([
        [0],
        [1/m]
    ])

    # Design cost matrices
    Q = np.array([
        [10, 0],   # High cost for position error
        [0, 1]     # Lower cost for velocity error
    ])

    R = np.array([[0.1]])  # Low cost for control effort

    # Create LQR controller
    lqr_controller = LQRController(A, B, Q, R)

    print(f"LQR gain matrix K: {lqr_controller.K}")
    print(f"Closed-loop poles: {np.linalg.eigvals(A - B @ lqr_controller.K)}")

    # Test with initial state
    initial_state = np.array([1.0, 0.0])  # Position = 1, Velocity = 0
    control_input = lqr_controller.control(initial_state)
    print(f"Control input for state {initial_state}: {control_input}")
```

### Feedback Linearization

```python
#!/usr/bin/env python3

import numpy as np
from typing import Callable, Tuple
import matplotlib.pyplot as plt

class FeedbackLinearizationController:
    def __init__(self, f: Callable, g: Callable, h: Callable,
                 linear_controller: Callable):
        """
        Feedback linearization controller
        For system: ẋ = f(x) + g(x)u, y = h(x)
        Transforms to linear system: ẏ = v (virtual input)
        :param f: Drift dynamics f(x)
        :param g: Control input matrix g(x)
        :param h: Output function h(x)
        :param linear_controller: Controller for linearized system v = k(e, ė)
        """
        self.f = f
        self.g = g
        self.h = h
        self.linear_controller = linear_controller

    def control(self, x: np.ndarray, y_ref: float,
                y_ref_dot: float = 0.0) -> float:
        """
        Compute control input using feedback linearization
        :param x: Current state
        :param y_ref: Reference output
        :param y_ref_dot: Reference output derivative
        :return: Control input u
        """
        # Calculate output and its derivatives
        y = self.h(x)
        Lfh = self._lie_derivative(self.h, self.f, x)
        Lgh = self._lie_derivative(self.h, self.g, x)

        # Calculate Lie derivatives of output
        y_dot = Lfh  # First derivative of output

        # Calculate virtual control input for linearized system
        e = y_ref - y
        e_dot = y_ref_dot - y_dot

        # Use linear controller to get virtual input
        v = self.linear_controller(e, e_dot)

        # Compute actual control input
        # v = Lf^2h(x) + LgLf h(x) * u
        # u = (v - Lf^2h(x)) / LgLf h(x)

        # For first-order system, we need the second derivative
        u = (v - Lfh) / Lgh if Lgh != 0 else 0.0

        return u

    def _lie_derivative(self, h: Callable, f: Callable, x: np.ndarray) -> float:
        """
        Calculate Lie derivative L_f h = ∇h * f
        """
        # Numerical gradient of h
        eps = 1e-8
        grad_h = np.zeros_like(x)
        h0 = h(x)

        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            grad_h[i] = (h(x_plus) - h0) / eps

        # Calculate f(x)
        if callable(f):
            f_val = f(x)
        else:
            f_val = f  # Assume f is already evaluated

        # Lie derivative
        return np.dot(grad_h, f_val)

class ComputedTorqueController:
    def __init__(self, M_func: Callable, C_func: Callable, G_func: Callable,
                 Kp: np.ndarray, Kd: np.ndarray):
        """
        Computed torque controller for robotic manipulators
        Implements: τ = M(q) * (q̈_d + Kp*e + Kd*ė) + C(q, q̇)*q̇ + G(q)
        :param M_func: Inertia matrix function M(q)
        :param C_func: Coriolis matrix function C(q, q̇)
        :param G_func: Gravity vector function G(q)
        :param Kp: Proportional gain matrix
        :param Kd: Derivative gain matrix
        """
        self.M_func = M_func
        self.C_func = C_func
        self.G_func = G_func
        self.Kp = Kp
        self.Kd = Kd

    def control(self, q: np.ndarray, q_dot: np.ndarray,
                q_desired: np.ndarray, qd_desired: np.ndarray, qdd_desired: np.ndarray) -> np.ndarray:
        """
        Compute control torque using computed torque method
        :param q: Current joint positions
        :param q_dot: Current joint velocities
        :param q_desired: Desired joint positions
        :param qd_desired: Desired joint velocities
        :param qdd_desired: Desired joint accelerations
        :return: Control torques
        """
        # Calculate tracking errors
        e = q_desired - q
        edot = qd_desired - q_dot

        # Calculate desired acceleration with PD feedback
        qdd_cmd = qdd_desired + self.Kp @ e + self.Kd @ edot

        # Calculate control torques
        M = self.M_func(q)
        C = self.C_func(q, q_dot)
        G = self.G_func(q)

        tau = M @ qdd_cmd + C @ q_dot + G

        return tau

class SlidingModeController:
    def __init__(self, A: np.ndarray, B: np.ndarray,
                 switching_gain: float = 1.0,
                 boundary_layer_thickness: float = 0.1):
        """
        Sliding mode controller
        :param A: System matrix
        :param B: Input matrix
        :param switching_gain: Gain for discontinuous control
        :param boundary_layer_thickness: Thickness of boundary layer to reduce chattering
        """
        self.A = A
        self.B = B
        self.rho = switching_gain
        self.phi = boundary_layer_thickness

        # Design sliding surface parameters (s = C*x where C makes A-BK Hurwitz)
        # For simplicity, we'll use a simple design
        n = A.shape[0]
        self.C = np.eye(n)  # Simple sliding surface

    def control(self, x: np.ndarray, reference: np.ndarray = None) -> np.ndarray:
        """
        Compute sliding mode control input
        :param x: Current state
        :param reference: Reference state
        :return: Control input
        """
        if reference is None:
            reference = np.zeros_like(x)

        # Calculate error
        e = reference - x

        # Calculate sliding surface
        s = self.C @ e

        # Calculate control input
        # u = u_eq + u_switch
        # u_switch = -rho * sign(s) (or sat(s/phi) to reduce chattering)

        # Equivalent control (assuming linear system)
        ueq = np.linalg.pinv(self.B) @ (self.A @ e)

        # Switching control with boundary layer
        switching_control = np.zeros_like(s)
        for i in range(len(s)):
            if abs(s[i]) <= self.phi:
                switching_control[i] = -self.rho * s[i] / self.phi  # Linear in boundary layer
            else:
                switching_control[i] = -self.rho * np.sign(s[i])   # Discontinuous outside

        # Total control
        u = ueq + switching_control

        return u

# Example: Feedback linearization for a simple pendulum
def pendulum_f(x):
    """Drift dynamics for pendulum: ẋ₁ = x₂, ẋ₂ = -g/l*sin(x₁) - c*x₂"""
    g, l, c = 9.81, 1.0, 0.1
    return np.array([x[1], -g/l * np.sin(x[0]) - c * x[1]])

def pendulum_g(x):
    """Control input matrix for pendulum: g = [0, 1/J]"""
    J = 0.1  # Moment of inertia
    return np.array([0, 1/J])

def pendulum_output(x):
    """Output function: angle"""
    return x[0]

# Example linear controller (PD controller)
def pd_controller(e, edot, kp=10.0, kd=2.0):
    return kp * e + kd * edot

if __name__ == "__main__":
    # Create feedback linearization controller for pendulum
    fl_controller = FeedbackLinearizationController(
        pendulum_f, pendulum_g, pendulum_output,
        lambda e, edot: pd_controller(e, edot)
    )

    # Test with initial state
    initial_state = np.array([0.1, 0.0])  # Small angle, zero velocity
    control_input = fl_controller.control(initial_state, 0.0)  # Control to zero angle
    print(f"Feedback linearization control input: {control_input}")

    # Create computed torque controller parameters
    n_joints = 3
    Kp = 10.0 * np.eye(n_joints)
    Kd = 2.0 * np.eye(n_joints)

    # Simple functions for robot dynamics (in practice, these would be complex)
    def M_func(q):
        return np.eye(n_joints)  # Identity for simplicity

    def C_func(q, qdot):
        return np.zeros((n_joints, n_joints))

    def G_func(q):
        return np.zeros(n_joints)

    ct_controller = ComputedTorqueController(M_func, C_func, G_func, Kp, Kd)

    # Test computed torque controller
    q = np.array([0.1, 0.2, 0.3])
    qdot = np.array([0.0, 0.0, 0.0])
    qd = np.array([0.0, 0.0, 0.0])
    qdd = np.array([0.0, 0.0, 0.0])

    torques = ct_controller.control(q, qdot, qd, qd, qdd)
    print(f"Computed torque controller output: {torques}")
```

## Trajectory Tracking Control

### Model Reference Adaptive Control (MRAC)

```python
#!/usr/bin/env python3

import numpy as np
from typing import Tuple, Callable
import matplotlib.pyplot as plt

class MRACController:
    def __init__(self, A: np.ndarray, B: np.ndarray,
                 Am: np.ndarray, Bm: np.ndarray,
                 gamma_theta: float = 0.1,
                 gamma_w: float = 0.1):
        """
        Model Reference Adaptive Control
        :param A: Plant system matrix
        :param B: Plant input matrix
        :param Am: Reference model system matrix
        :param Bm: Reference model input matrix
        :param gamma_theta: Adaptation rate for controller parameters
        :param gamma_w: Adaptation rate for system parameters
        """
        self.A = A
        self.B = B
        self.Am = Am
        self.Bm = Bm

        # System dimensions
        self.n = A.shape[0]  # State dimension
        self.m = B.shape[1]  # Input dimension

        # Adaptation rates
        self.gamma_theta = gamma_theta
        self.gamma_w = gamma_w

        # Initialize adaptive parameters
        self.theta = np.zeros((self.m, self.n))  # Controller gain
        self.w = np.zeros((self.n, self.n))      # System parameter estimate

        # Positive definite matrices for Lyapunov function
        self.Q = np.eye(self.n)  # Can be tuned
        self.P = np.linalg.solve_continuous_lyapunov(
            Am.T, -self.Q
        )  # Solution to Am^T*P + P*Am = -Q

    def control(self, x: np.ndarray, r: np.ndarray) -> np.ndarray:
        """
        Compute control input using MRAC
        :param x: Current system state
        :param r: Reference input
        :return: Control input
        """
        # Reference model state
        xm = self.Am @ x + self.Bm @ r

        # Tracking error
        e = x - xm

        # Control law: u = -theta * x + ur (ur is reference input)
        u = -self.theta @ x + r

        # Adaptation laws
        # dθ/dt = -gamma_theta * P * B^T * e
        # dw/dt = -gamma_w * e * x^T
        dtheta = -self.gamma_theta * e @ self.P @ self.B @ x.reshape(-1, 1)
        dtheta = dtheta.flatten()

        # Update parameters
        self.theta += dtheta.reshape(self.theta.shape) * 0.01  # Integration step
        self.w += (-self.gamma_w * np.outer(e, x)) * 0.01

        return u

class RobustController:
    def __init__(self, A: np.ndarray, B: np.ndarray,
                 Q: np.ndarray, R: np.ndarray,
                 uncertainty_bound: float = 1.0):
        """
        Robust controller with uncertainty compensation
        :param A: Nominal system matrix
        :param B: Nominal input matrix
        :param Q: State cost matrix
        :param R: Input cost matrix
        :param uncertainty_bound: Bound on system uncertainty
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.uncertainty_bound = uncertainty_bound

        # Design nominal LQR controller
        self.lqr_gain = self._design_lqr(A, B, Q, R)

        # Robustness parameter
        self.kappa = 1.5  # Scaling factor for robust control

    def _design_lqr(self, A: np.ndarray, B: np.ndarray,
                   Q: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Design LQR controller for nominal system"""
        P = np.linalg.solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ B.T @ P
        return K

    def control(self, x: np.ndarray, reference: np.ndarray = None) -> np.ndarray:
        """
        Compute robust control input
        :param x: Current state
        :param reference: Reference state
        :return: Control input
        """
        if reference is None:
            reference = np.zeros_like(x)

        # Nominal control (LQR)
        error = reference - x
        u_nominal = self.lqr_gain @ error

        # Robust control component to handle uncertainties
        # u_robust = -kappa * uncertainty_bound * ||error|| * sign(error)
        norm_error = np.linalg.norm(error)
        if norm_error > 1e-6:  # Avoid division by zero
            u_robust = -self.kappa * self.uncertainty_bound * norm_error * (error / norm_error)
        else:
            u_robust = np.zeros_like(x)

        # Total control
        u_total = u_nominal + u_robust

        return u_total

class AdaptiveRobustController:
    def __init__(self, n_states: int, n_inputs: int,
                 lambda_adapt: float = 0.01,
                 robust_gain: float = 1.0):
        """
        Adaptive robust controller combining adaptation and robustness
        :param n_states: Number of system states
        :param n_inputs: Number of control inputs
        :param lambda_adapt: Adaptation rate
        :param robust_gain: Robust control gain
        """
        self.n_states = n_states
        self.n_inputs = n_inputs
        self.lambda_adapt = lambda_adapt
        self.robust_gain = robust_gain

        # Adaptive parameters (for system identification)
        self.theta = np.zeros(n_states * n_states + n_states * n_inputs)  # Vectorized parameters
        self.P = np.eye(len(self.theta)) * 10  # Covariance matrix

    def regressor(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Regressor vector for system identification
        For system: ẋ = f(x) + g(x)u, we estimate f and g
        """
        # Concatenate x and u to form regressor
        phi = np.concatenate([x, np.kron(x, u)])  # Example regressor structure
        return phi

    def control(self, x: np.ndarray, x_ref: np.ndarray,
                dt: float = 0.01) -> np.ndarray:
        """
        Compute adaptive robust control
        """
        # Simple PD controller part
        e = x_ref - x
        Kp = 5.0 * np.eye(self.n_states)
        Kd = 2.0 * np.eye(self.n_states)

        # Estimate system parameters (simplified)
        # In practice, this would use the adaptive law
        u_adaptive = Kp @ e  # PD control part

        # Robust control part to handle parameter uncertainties
        norm_e = np.linalg.norm(e)
        if norm_e > 1e-6:
            u_robust = -self.robust_gain * norm_e * (e / norm_e)
        else:
            u_robust = np.zeros_like(e)

        # Total control
        u_total = u_adaptive + u_robust

        return u_total

# Example: Test controllers with a simple system
if __name__ == "__main__":
    # Define a simple 2nd order system
    A = np.array([[0, 1], [-2, -1]])  # Stable system
    B = np.array([[0], [1]])

    # Reference model (desired closed-loop dynamics)
    Am = np.array([[0, 1], [-4, -2]])  # Faster response
    Bm = B  # Same input matrix

    # Cost matrices for LQR
    Q = np.eye(2)
    R = np.array([[1]])

    # Create controllers
    mrac = MRACController(A, B, Am, Bm)
    robust = RobustController(A, B, Q, R, uncertainty_bound=0.5)

    print("Controllers initialized successfully")

    # Test with initial conditions
    x0 = np.array([1.0, 0.5])
    reference = np.array([0.0, 0.0])

    u_robust = robust.control(x0, reference)
    print(f"Robust control input: {u_robust}")
```

## Practical Control Implementations

### Real-time Control Architecture

```python
#!/usr/bin/env python3

import numpy as np
import time
import threading
from typing import Callable, Dict, Any
import queue
import signal
import sys

class RealTimeController:
    def __init__(self, control_frequency: float = 100.0):
        """
        Real-time controller with timing guarantees
        :param control_frequency: Control loop frequency in Hz
        """
        self.control_frequency = control_frequency
        self.dt = 1.0 / control_frequency
        self.running = False
        self.control_thread = None

        # Control components
        self.current_state = np.array([])
        self.desired_state = np.array([])
        self.control_output = np.array([])

        # Timing statistics
        self.loop_times = []
        self.target_time = 0.0
        self.actual_time = 0.0

        # Controller function
        self.controller_func: Callable = None

    def set_controller(self, controller_func: Callable):
        """Set the control function"""
        self.controller_func = controller_func

    def set_state(self, state: np.ndarray):
        """Set current system state"""
        self.current_state = state

    def set_desired_state(self, desired_state: np.ndarray):
        """Set desired system state"""
        self.desired_state = desired_state

    def start(self):
        """Start the real-time control loop"""
        if self.control_thread is not None and self.control_thread.is_alive():
            return

        self.running = True
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.start()

    def stop(self):
        """Stop the real-time control loop"""
        self.running = False
        if self.control_thread is not None:
            self.control_thread.join()

    def _control_loop(self):
        """Main control loop with real-time timing"""
        next_loop_time = time.time()

        while self.running:
            loop_start = time.time()

            try:
                # Execute control algorithm
                if self.controller_func is not None:
                    self.control_output = self.controller_func(
                        self.current_state, self.desired_state
                    )

                # Update timing statistics
                loop_time = time.time() - loop_start
                self.loop_times.append(loop_time)

                if len(self.loop_times) > 1000:  # Keep last 1000 samples
                    self.loop_times.pop(0)

                # Calculate sleep time to maintain target frequency
                next_loop_time += self.dt
                sleep_time = next_loop_time - time.time()

                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    # Missed deadline, log warning
                    print(f"Control loop missed deadline by {-sleep_time:.4f}s")

            except Exception as e:
                print(f"Error in control loop: {e}")
                time.sleep(0.001)  # Brief pause to avoid busy loop on error

    def get_timing_stats(self) -> Dict[str, float]:
        """Get timing statistics"""
        if not self.loop_times:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}

        times = np.array(self.loop_times)
        return {
            "mean": np.mean(times),
            "std": np.std(times),
            "min": np.min(times),
            "max": np.max(times),
            "frequency": 1.0 / np.mean(times) if np.mean(times) > 0 else 0
        }

class HierarchicalController:
    def __init__(self):
        """
        Hierarchical control system with multiple levels:
        - Trajectory planning (high level)
        - Path following (mid level)
        - Low-level control (low level)
        """
        self.trajectory_planner = None
        self.path_follower = None
        self.low_level_controller = None

        # State variables
        self.current_position = np.zeros(2)
        self.current_velocity = np.zeros(2)
        self.desired_trajectory = []
        self.current_reference = np.zeros(2)

    def set_trajectory_planner(self, planner_func: Callable):
        """Set trajectory planning function"""
        self.trajectory_planner = planner_func

    def set_path_follower(self, follower_func: Callable):
        """Set path following function"""
        self.path_follower = follower_func

    def set_low_level_controller(self, controller_func: Callable):
        """Set low-level control function"""
        self.low_level_controller = controller_func

    def update(self, dt: float):
        """Update hierarchical control system"""
        # High-level: trajectory planning
        if self.trajectory_planner:
            self.desired_trajectory = self.trajectory_planner()

        # Mid-level: path following
        if self.path_follower and len(self.desired_trajectory) > 0:
            self.current_reference = self.path_follower(
                self.current_position, self.desired_trajectory
            )

        # Low-level: execute control
        if self.low_level_controller:
            control_output = self.low_level_controller(
                self.current_position, self.current_velocity,
                self.current_reference
            )
            return control_output

        return np.zeros(2)  # Default control

class SafetyController:
    def __init__(self, soft_limits: Dict[str, tuple], hard_limits: Dict[str, tuple]):
        """
        Safety controller to enforce system limits
        :param soft_limits: Dict of soft limits {'position': (min, max), ...}
        :param hard_limits: Dict of hard limits that cause emergency stop
        """
        self.soft_limits = soft_limits
        self.hard_limits = hard_limits
        self.emergency_stop = False
        self.warning_issued = False

    def check_limits(self, state: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if system is within limits
        :return: (is_safe, warning_message)
        """
        if self.emergency_stop:
            return False, "Emergency stop active"

        for var_name, value in state.items():
            # Check hard limits
            if var_name in self.hard_limits:
                min_val, max_val = self.hard_limits[var_name]
                if value < min_val or value > max_val:
                    self.emergency_stop = True
                    return False, f"Hard limit exceeded for {var_name}: {value} not in [{min_val}, {max_val}]"

            # Check soft limits
            if var_name in self.soft_limits:
                min_val, max_val = self.soft_limits[var_name]
                if value < min_val or value > max_val:
                    if not self.warning_issued:
                        self.warning_issued = True
                        return False, f"Soft limit exceeded for {var_name}: {value} not in [{min_val}, {max_val}]"

        return True, ""

    def reset(self):
        """Reset safety controller"""
        self.emergency_stop = False
        self.warning_issued = False

# Example: Simple robot control system
class RobotControlSystem:
    def __init__(self):
        # Initialize controllers
        self.pid_controller = MultiDOFPIDController(2, 2.0, 0.1, 0.05)  # 2 DOF robot
        self.real_time_ctrl = RealTimeController(100)  # 100 Hz control
        self.safety_ctrl = SafetyController(
            soft_limits={'position_x': (-10, 10), 'position_y': (-10, 10)},
            hard_limits={'position_x': (-15, 15), 'position_y': (-15, 15)}
        )

        # Robot state
        self.current_pos = np.array([0.0, 0.0])
        self.current_vel = np.array([0.0, 0.0])
        self.desired_pos = np.array([1.0, 1.0])

    def control_step(self):
        """Execute one control step"""
        # Check safety
        state_check = {
            'position_x': self.current_pos[0],
            'position_y': self.current_pos[1]
        }

        is_safe, warning = self.safety_ctrl.check_limits(state_check)
        if not is_safe:
            print(f"Safety warning: {warning}")
            return np.array([0.0, 0.0])  # Zero control if unsafe

        # Execute PID control
        control_output = self.pid_controller.update(
            self.current_pos.tolist(),
            self.desired_pos.tolist(),
            dt=0.01
        )

        # Update robot simulation (simplified)
        self.current_vel = np.array(control_output) * 0.1  # Simple integration
        self.current_pos += self.current_vel * 0.01

        return np.array(control_output)

if __name__ == "__main__":
    # Create robot control system
    robot_ctrl = RobotControlSystem()

    # Run control loop for a few steps
    for i in range(10):
        control_output = robot_ctrl.control_step()
        print(f"Step {i}: Position={robot_ctrl.current_pos}, Control={control_output}")
        time.sleep(0.01)  # Simulate time delay

    print("Robot control system test completed")
```

## Practical Exercises

### Exercise 1: Implement a PID Tuning Algorithm

**Objective**: Create an automatic PID tuning system using the Ziegler-Nichols method or other tuning techniques.

**Steps**:
1. Implement system identification to determine plant characteristics
2. Apply Ziegler-Nichols or other tuning rules
3. Test the tuned controller with different systems
4. Compare performance with manually tuned parameters
5. Create visualization tools to show tuning process

**Expected Outcome**: An automatic PID tuning system that can identify system characteristics and provide well-tuned parameters.

### Exercise 2: Design a Hierarchical Control System

**Objective**: Implement a complete hierarchical control system for a mobile robot.

**Steps**:
1. Create high-level trajectory planner
2. Implement mid-level path following controller
3. Design low-level motor control
4. Integrate safety and monitoring systems
5. Test with realistic simulation

**Expected Outcome**: A working hierarchical control system that demonstrates the integration of multiple control levels.

### Exercise 3: Robust Control Design

**Objective**: Design and implement a robust controller that can handle system uncertainties.

**Steps**:
1. Model system uncertainties
2. Design robust controller using H-infinity or other methods
3. Implement adaptive components
4. Test with various uncertainty levels
5. Analyze performance and stability margins

**Expected Outcome**: A robust control system that maintains performance despite model uncertainties.

## Chapter Summary

This chapter covered the essential control systems used in robotics:

1. **Classical Control**: PID controllers and their advanced variants with anti-windup, filtering, and adaptive capabilities.

2. **Modern Control**: LQR, MPC, and feedback linearization techniques for optimal and systematic control design.

3. **Advanced Methods**: Computed torque control, sliding mode control, and adaptive control for handling complex robotic dynamics.

4. **Real-time Implementation**: Practical considerations for implementing control systems with timing constraints and safety.

5. **Hierarchical Control**: Multi-level control architectures that coordinate planning, path following, and low-level control.

The choice of control strategy depends on the specific requirements of the robotic application, including accuracy, speed, robustness, and computational constraints. Modern robotics increasingly relies on sophisticated control techniques that can handle uncertainties, nonlinearities, and real-time constraints.

## Further Reading

1. "Feedback Control of Dynamic Systems" by Franklin et al. - Comprehensive control theory text
2. "Robotics: Control, Sensing, Vision, and Intelligence" by Fu et al. - Robotics-specific control
3. "Applied Nonlinear Control" by Slotine and Li - Advanced nonlinear control techniques
4. "Model Predictive Control" by Maciejowski - In-depth MPC theory and implementation
5. "Adaptive Control" by Astrom and Wittenmark - Adaptive control systems

## Assessment Questions

1. Derive the PID control equation and explain the role of each term in system response.

2. Implement a LQR controller for a 2-DOF robotic manipulator and analyze its performance.

3. Compare feedback linearization with computed torque control for robotic systems.

4. Design a sliding mode controller for a mobile robot and prove its stability.

5. Explain the differences between model reference adaptive control and self-tuning regulators.

6. Implement a model predictive controller for trajectory tracking with constraints.

7. Analyze the robustness properties of different control strategies for uncertain systems.

8. Design a hierarchical control system for a complex robotic application.

9. Evaluate the computational requirements of different control algorithms for real-time implementation.

10. Discuss the trade-offs between performance and robustness in control system design.

