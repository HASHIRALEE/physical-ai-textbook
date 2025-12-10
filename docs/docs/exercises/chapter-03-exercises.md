---
sidebar_position: 3
---

# Chapter 3 Exercises: Perception Systems

## Exercise 1: Stereo Vision Implementation

### Problem
Implement a stereo vision system to calculate depth from two camera images:
1. Load stereo images (left and right)
2. Compute disparity map
3. Convert disparity to depth
4. Visualize the 3D point cloud

### Solution Template
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def compute_disparity_map(img_left, img_right):
    """
    Compute disparity map using OpenCV's StereoSGBM
    """
    # Convert to grayscale if needed
    if len(img_left.shape) == 3:
        img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    if len(img_right.shape) == 3:
        img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # Create stereo matcher
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=64,  # Must be divisible by 16
        blockSize=11,
        P1=8 * 3 * 11**2,
        P2=32 * 3 * 11**2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # Compute disparity
    disparity = stereo.compute(img_left, img_right).astype(np.float32) / 16.0

    return disparity

def disparity_to_depth(disparity, baseline=0.2, focal_length=800):
    """
    Convert disparity to depth
    baseline: distance between cameras (m)
    focal_length: camera focal length in pixels
    """
    # Avoid division by zero
    disparity[disparity == 0] = 0.01
    depth = (baseline * focal_length) / disparity
    return depth

def visualize_point_cloud(depth_map, img_left, max_depth=10.0):
    """
    Visualize 3D point cloud from depth map
    """
    h, w = depth_map.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # Filter out points that are too far
    valid = (depth_map > 0) & (depth_map < max_depth)

    x_valid = x[valid].flatten()
    y_valid = y[valid].flatten()
    z_valid = depth_map[valid].flatten()

    # Convert pixel coordinates to 3D world coordinates (simplified)
    fx, fy = 800, 800  # focal lengths
    cx, cy = w/2, h/2  # principal points

    X = (x_valid - cx) * z_valid / fx
    Y = (y_valid - cy) * z_valid / fy
    Z = z_valid

    # Create 3D plot
    fig = plt.figure(figsize=(12, 5))

    # Show original image
    plt.subplot(1, 2, 1)
    plt.imshow(img_left, cmap='gray')
    plt.title('Left Image')
    plt.axis('off')

    # Show point cloud
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.scatter(X[::100], Y[::100], Z[::100], c=Z[::100], cmap='viridis', s=1)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Point Cloud')

    plt.tight_layout()
    plt.show()

# Example usage (with simulated images)
# In practice, you would load real stereo images
img_left = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
img_right = np.random.randint(0, 255, (480, 640), dtype=np.uint8)

# Add some pattern to make disparity calculation meaningful
for i in range(0, 480, 50):
    for j in range(0, 640, 50):
        cv2.rectangle(img_left, (j, i), (j+40, i+40), 255, -1)
        cv2.rectangle(img_right, (max(0, j-5), i), (max(0, j-5)+40, i+40), 255, -1)

disparity = compute_disparity_map(img_left, img_right)
depth = disparity_to_depth(disparity)

print(f"Disparity range: {disparity.min():.2f} to {disparity.max():.2f}")
print(f"Depth range: {depth.min():.2f} to {depth.max():.2f} meters")

visualize_point_cloud(depth, img_left)
```

## Exercise 2: Sensor Fusion with Extended Kalman Filter

### Problem
Implement an Extended Kalman Filter (EKF) for fusing visual and IMU data:
1. Model the humanoid's state (position, velocity, orientation)
2. Implement prediction and update steps
3. Test with simulated data

### Implementation Tasks:
1. Define state vector [x, y, z, vx, vy, vz, qx, qy, qz, qw]
2. Implement motion model using IMU data
3. Implement measurement model for visual observations
4. Test convergence and stability

### Solution Template
```python
import numpy as np

class ExtendedKalmanFilter:
    def __init__(self, dt=0.01):
        # State: [x, y, z, vx, vy, vz, qx, qy, qz, qw] (10 elements)
        n_states = 10
        n_measurements = 7  # [x, y, z, qx, qy, qz, qw]

        self.dt = dt
        self.state = np.zeros(n_states)
        self.state[6] = 1  # Initialize quaternion to [0, 0, 0, 1]

        # Covariance matrix
        self.P = np.eye(n_states) * 1.0

        # Process noise
        self.Q = np.eye(n_states) * 0.1

        # Measurement noise
        self.R = np.eye(n_measurements) * 0.5

    def predict(self, control_input=None):
        """
        Predict next state using motion model
        control_input: [ax, ay, az, wx, wy, wz] (acceleration and angular velocity)
        """
        if control_input is not None:
            ax, ay, az, wx, wy, wz = control_input

            # Update velocities
            self.state[3] += ax * self.dt  # vx
            self.state[4] += ay * self.dt  # vy
            self.state[5] += az * self.dt  # vz

            # Update positions
            self.state[0] += self.state[3] * self.dt  # x
            self.state[1] += self.state[4] * self.dt  # y
            self.state[2] += self.state[5] * self.dt  # z

            # Update orientation (simplified quaternion integration)
            q = self.state[6:10]
            omega = np.array([wx, wy, wz, 0])  # Angular velocity as quaternion
            q_dot = 0.5 * self.quat_multiply(omega, q)
            self.state[6:10] += q_dot * self.dt

            # Normalize quaternion
            self.state[6:10] /= np.linalg.norm(self.state[6:10])

        # Jacobian of motion model (simplified)
        F = np.eye(len(self.state))
        # Add appropriate Jacobian entries based on your motion model

        # Update covariance
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurement):
        """
        Update state with measurement
        measurement: [x, y, z, qx, qy, qz, qw]
        """
        # Measurement function (identity for position/orientation)
        H = np.zeros((len(measurement), len(self.state)))
        H[0, 0] = 1  # x
        H[1, 1] = 1  # y
        H[2, 2] = 1  # z
        H[3, 6] = 1  # qx
        H[4, 7] = 1  # qy
        H[5, 8] = 1  # qz
        H[6, 9] = 1  # qw

        # Innovation
        y = measurement - self.state[:len(measurement)]

        # Innovation covariance
        S = H @ self.P @ H.T + self.R

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state
        self.state = self.state + K @ y

        # Update covariance
        I = np.eye(len(self.state))
        self.P = (I - K @ H) @ self.P

    def quat_multiply(self, q1, q2):
        """
        Multiply two quaternions
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        return np.array([w, x, y, z])

# Test the EKF
ekf = ExtendedKalmanFilter(dt=0.01)

# Simulate measurements
for t in np.arange(0, 5, 0.01):
    # Simulate motion
    control = [0.1, 0.05, 0, 0.01, 0.005, 0.002]  # Acceleration and angular velocity
    ekf.predict(control_input=control)

    # Simulate measurements (with noise)
    true_pos = [0.1*t**2, 0.05*t**2, 0]  # True position
    true_quat = [0.01*t, 0.005*t, 0.002*t, 1]  # True orientation
    true_quat = np.array(true_quat) / np.linalg.norm(true_quat)

    measurement = np.concatenate([
        np.array(true_pos) + np.random.normal(0, 0.01, 3),  # Position + noise
        true_quat + np.random.normal(0, 0.001, 4)  # Orientation + noise
    ])

    ekf.update(measurement[:7])  # Use only first 7 elements (pos + quat)

print(f"Final state: {ekf.state}")
print(f"Final covariance trace: {np.trace(ekf.P)}")
```

## Exercise 3: Tactile Pattern Recognition

### Problem
Create a system that recognizes objects by touch using tactile sensors:
1. Generate synthetic tactile data for different objects
2. Train a classifier to recognize different textures/materials
3. Test the system with new tactile inputs

### Implementation Tasks:
1. Create tactile data generator for different textures
2. Implement a neural network classifier
3. Evaluate classification performance

## Exercise 4: ROS2 Perception Pipeline

### Problem
Create a complete ROS2 perception pipeline:
1. Subscribe to multiple sensor topics
2. Implement sensor fusion
3. Publish processed perception data
4. Add visualization tools

### Requirements:
1. Handle sensor synchronization
2. Implement error handling for missing data
3. Add performance monitoring
4. Create launch file for the pipeline

## Challenge Exercise: Active Perception

### Problem
Implement an active perception system that controls the robot's head/camera to improve object recognition:
1. Plan head movements to gather more information
2. Implement uncertainty reduction strategies
3. Balance between perception and other tasks

### Considerations:
- How does the robot decide where to look?
- How to integrate active perception with navigation?
- How to handle competing objectives (perception vs. balance)?