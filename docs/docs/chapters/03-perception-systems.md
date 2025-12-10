---
id: ch03-perception-systems
title: Perception Systems
sidebar_label: "Chapter 3: Perception Systems"
sidebar_position: 3
description: Understanding the different sensory modalities used in humanoid robots
---

# Chapter 3: Perception Systems

## Learning Objectives

By the end of this chapter, students will be able to:
- Understand the different sensory modalities used in humanoid robots
- Implement basic computer vision algorithms for humanoid perception
- Design sensor fusion systems for robust perception
- Apply machine learning techniques to sensor data
- Integrate perception with action in humanoid robots

## Theoretical Foundations

### Sensory Modalities in Humanoid Robots

Humanoid robots employ multiple sensory systems to perceive their environment:

1. **Vision Systems**: Cameras for color, depth, and motion information
2. **Tactile Sensing**: Force/torque sensors, tactile skin for contact detection
3. **Proprioception**: Joint encoders, IMU sensors for self-awareness
4. **Auditory Systems**: Microphones for sound and speech recognition
5. **Range Sensing**: LIDAR, ultrasonic sensors for distance measurement

### Computer Vision for Humanoids

Computer vision systems in humanoid robots face unique challenges:

- **Ego-motion compensation**: The robot's own movement affects visual input
- **Active vision**: Robots can move their cameras to improve perception
- **Real-time processing**: Limited computational resources require efficient algorithms
- **Multi-camera fusion**: Combining information from multiple viewpoints

### Sensor Fusion

Sensor fusion combines data from multiple sensors to improve perception reliability:

- **Kalman Filtering**: Optimal estimation for linear systems with Gaussian noise
- **Particle Filtering**: Non-parametric approach for non-linear, non-Gaussian systems
- **Bayesian Networks**: Probabilistic reasoning with multiple sensor inputs
- **Deep Learning Fusion**: Neural networks that learn to combine sensor data

## Practical Examples

### Example 1: Visual Object Detection for Humanoids

```python
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Point

class HumanoidObjectDetector:
    def __init__(self):
        self.bridge = CvBridge()
        self.object_pub = rospy.Publisher('/detected_objects', Point, queue_size=10)

        # Load pre-trained model (simplified example)
        self.net = cv2.dnn.readNetFromDarknet('yolo.cfg', 'yolo.weights')
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect_objects(self, image):
        """
        Detect objects in an image using YOLO
        """
        height, width, channels = image.shape

        # Prepare image for detection
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        # Process detections
        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:  # Confidence threshold
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-maximum suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        return boxes, confidences, class_ids, indexes

    def process_image(self, img_msg):
        """
        Process ROS image message and detect objects
        """
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        boxes, confidences, class_ids, indexes = self.detect_objects(cv_image)

        # Draw detections on image
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(boxes), 3))

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(class_ids[i])
                confidence = confidences[i]

                cv2.rectangle(cv_image, (x, y), (x + w, y + h), colors[i], 2)
                cv2.putText(cv_image, f"{label} {confidence:.2f}", (x, y + 30), font, 2, colors[i], 3)

        # Publish object positions (simplified)
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                obj_point = Point()
                obj_point.x = x + w/2  # Center x
                obj_point.y = y + h/2  # Center y
                obj_point.z = confidences[i]  # Use confidence as z for simplicity
                self.object_pub.publish(obj_point)
```

### Example 2: Sensor Fusion with Kalman Filter

```python
import numpy as np

class KalmanFilter:
    def __init__(self, dt, n_state, n_measurement):
        self.dt = dt
        self.n_state = n_state
        self.n_measurement = n_measurement

        # State: [x, y, z, vx, vy, vz]
        self.state = np.zeros(n_state)  # Initial state

        # Uncertainty covariance
        self.P = np.eye(n_state) * 1000  # Initial uncertainty

        # Process noise covariance
        self.Q = np.eye(n_state) * 0.1

        # Measurement noise covariance
        self.R = np.eye(n_measurement) * 10

        # State transition model
        self.F = np.eye(n_state)
        # For constant velocity model
        for i in range(3):
            self.F[i, i+3] = dt

        # Measurement function
        self.H = np.zeros((n_measurement, n_state))
        # For position-only measurements
        for i in range(min(n_measurement, 3)):
            self.H[i, i] = 1

    def predict(self):
        """Predict next state"""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement):
        """Update state with measurement"""
        # Calculate Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        y = measurement - self.H @ self.state  # Innovation
        self.state = self.state + K @ y

        # Update uncertainty
        I = np.eye(len(self.state))
        self.P = (I - K @ self.H) @ self.P

class HumanoidPerceptionFusion:
    def __init__(self):
        # 6-state Kalman filter: [x, y, z, vx, vy, vz]
        self.kf = KalmanFilter(dt=0.01, n_state=6, n_measurement=3)
        self.last_time = rospy.Time.now().to_sec()

    def fuse_sensors(self, vision_data, imu_data, joint_data):
        """
        Fuse data from multiple sensors
        """
        current_time = rospy.Time.now().to_sec()
        dt = current_time - self.last_time
        self.last_time = current_time

        # Update Kalman filter parameters
        self.kf.dt = dt
        self.kf.F = np.eye(6)
        for i in range(3):
            self.kf.F[i, i+3] = dt

        # Predict state
        self.kf.predict()

        # Fuse measurements
        # Vision provides position
        vision_pos = np.array([vision_data.x, vision_data.y, vision_data.z])

        # IMU provides acceleration (double integrate for position)
        # This is simplified - in practice, you'd use proper integration

        # Update with vision measurement
        self.kf.update(vision_pos)

        return self.kf.state[:3]  # Return position estimate
```

### Example 3: Tactile Perception

```python
import numpy as np

class TactilePerception:
    def __init__(self, n_sensors=100):
        self.n_sensors = n_sensors
        self.tactile_data = np.zeros(n_sensors)
        self.contact_threshold = 5.0  # Threshold for contact detection

    def detect_contact(self, tactile_readings):
        """
        Detect contact points from tactile sensor array
        """
        self.tactile_data = tactile_readings
        contacts = tactile_readings > self.contact_threshold

        # Find contact clusters
        contact_regions = []
        in_contact = False
        start_idx = 0

        for i, is_contact in enumerate(contacts):
            if is_contact and not in_contact:
                # Start of contact region
                in_contact = True
                start_idx = i
            elif not is_contact and in_contact:
                # End of contact region
                in_contact = False
                contact_regions.append((start_idx, i-1))

        # Handle case where contact continues to end
        if in_contact:
            contact_regions.append((start_idx, len(contacts)-1))

        return contact_regions

    def estimate_contact_force(self, contact_regions):
        """
        Estimate total contact force and center of pressure
        """
        total_force = 0.0
        cop_x = 0.0

        for start, end in contact_regions:
            region_force = np.sum(self.tactile_data[start:end+1])
            total_force += region_force

            # Calculate center of pressure for this region
            region_cop = np.average(
                range(start, end+1),
                weights=self.tactile_data[start:end+1]
            )
            cop_x += region_force * region_cop

        if total_force > 0:
            cop_x /= total_force
        else:
            cop_x = -1  # No contact

        return total_force, cop_x

# Example usage
tactile = TactilePerception(n_sensors=100)
simulated_readings = np.zeros(100)
# Simulate contact on sensors 30-40
simulated_readings[30:41] = np.random.normal(15, 2, 11)
simulated_readings[70:75] = np.random.normal(12, 1, 5)

contact_regions = tactile.detect_contact(simulated_readings)
total_force, cop_x = tactile.estimate_contact_force(contact_regions)

print(f"Contact regions: {contact_regions}")
print(f"Total force: {total_force:.2f}")
print(f"Center of pressure: {cop_x:.2f}")
```

## ROS2 Integration Example

Here's a ROS2 node for perception processing:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, Imu, JointState
from geometry_msgs.msg import PointStamped, PoseArray
from std_msgs.msg import Float32MultiArray
import cv2
from cv_bridge import CvBridge
import numpy as np

class HumanoidPerception(Node):
    def __init__(self):
        super().__init__('humanoid_perception')

        # Publishers
        self.object_pub = self.create_publisher(
            PoseArray,
            '/detected_objects',
            10
        )

        self.fused_state_pub = self.create_publisher(
            PointStamped,
            '/fused_position',
            10
        )

        self.tactile_pub = self.create_publisher(
            Float32MultiArray,
            '/tactile_processed',
            10
        )

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )

        # Timer for fusion loop
        self.timer = self.create_timer(0.01, self.fusion_loop)

        # Initialize processing components
        self.bridge = CvBridge()
        self.kalman_filter = KalmanFilter(dt=0.01, n_state=6, n_measurement=3)
        self.tactile_perception = TactilePerception(n_sensors=100)

        # Store sensor data
        self.latest_image = None
        self.latest_imu = None
        self.latest_joints = None
        self.has_data = False

    def image_callback(self, msg):
        """Process incoming image data"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # Store for processing in fusion loop
            self.latest_image = cv_image
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def imu_callback(self, msg):
        """Process incoming IMU data"""
        self.latest_imu = msg

    def joint_callback(self, msg):
        """Process incoming joint data"""
        self.latest_joints = msg

    def detect_objects_in_image(self, image):
        """Detect objects in image - simplified version"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Simple blob detection for demonstration
        detector = cv2.SimpleBlobDetector_create()
        keypoints = detector.detect(gray)

        # Extract object positions
        objects = []
        for kp in keypoints:
            obj = PointStamped()
            obj.point.x = kp.pt[0]  # Pixel x
            obj.point.y = kp.pt[1]  # Pixel y
            obj.point.z = 1.0       # Simulated depth
            objects.append(obj)

        return objects

    def fusion_loop(self):
        """Main fusion and processing loop"""
        if self.latest_image is not None:
            # Process vision data
            detected_objects = self.detect_objects_in_image(self.latest_image)

            # Publish object positions
            if detected_objects:
                obj_array = PoseArray()
                for obj in detected_objects:
                    obj_array.poses.append(obj.pose)
                self.object_pub.publish(obj_array)

        # Process IMU data for fusion
        if self.latest_imu is not None:
            # Extract orientation from IMU (simplified)
            imu_orientation = [
                self.latest_imu.orientation.x,
                self.latest_imu.orientation.y,
                self.latest_imu.orientation.z,
                self.latest_imu.orientation.w
            ]

            # For position fusion, we'd need to integrate acceleration
            # This is simplified for the example

            # Update Kalman filter
            self.kalman_filter.predict()

            # If we had position measurements from vision, we would update:
            # self.kalman_filter.update(position_measurement)

        # Publish fused state
        fused_pos = PointStamped()
        fused_pos.point.x = self.kalman_filter.state[0]
        fused_pos.point.y = self.kalman_filter.state[1]
        fused_pos.point.z = self.kalman_filter.state[2]
        self.fused_state_pub.publish(fused_pos)

def main(args=None):
    rclpy.init(args=args)
    perception_node = HumanoidPerception()
    rclpy.spin(perception_node)
    perception_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Interactive Exercises

### Exercise 1: Multi-camera Fusion
1. Implement a system that combines data from stereo cameras
2. Calculate depth information from stereo disparities
3. Integrate depth data with other sensors

### Exercise 2: SLAM Implementation
1. Implement a basic SLAM algorithm for humanoid navigation
2. Use visual and IMU data to build a map
3. Test the system in a simulated environment

### Exercise 3: Tactile Learning
1. Create a tactile perception system that learns to identify objects by touch
2. Use machine learning to classify different materials
3. Integrate tactile perception with vision for object identification

## Summary

This chapter covered perception systems in humanoid robots, including:

- Multiple sensory modalities used in humanoid robots
- Computer vision techniques for object detection and recognition
- Sensor fusion methods for combining information from multiple sensors
- Tactile perception for contact and force sensing
- ROS2 integration for real-time perception processing

Perception systems are crucial for humanoid robots to understand and interact with their environment. The integration of multiple sensors through fusion techniques enables robust and reliable perception capabilities.

## References and Further Reading

1. Thrun, S., Burgard, W., & Fox, D. (2005). Probabilistic Robotics.
2. Szeliski, R. (2010). Computer Vision: Algorithms and Applications.
3. Dahiya, R. S. (2013). Tactile sensing: From humans to humanoids.