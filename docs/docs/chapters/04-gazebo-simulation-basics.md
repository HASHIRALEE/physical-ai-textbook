---
sidebar_label: 'Chapter 4: Gazebo Simulation Basics'
title: 'Chapter 4: Gazebo Simulation Basics'
sidebar_position: 4
description: 'Learn the fundamentals of Gazebo simulation for robotics development'
---

# Chapter 4: Gazebo Simulation Basics

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand the fundamentals of Gazebo simulation environment
- Set up and configure Gazebo for robotics projects
- Create and manipulate robot models in Gazebo
- Implement basic simulation scenarios and experiments
- Integrate Gazebo with ROS2 for robot control

## Theoretical Foundations

### Introduction to Gazebo

Gazebo is a powerful open-source robotics simulator that provides realistic physics simulation, high-quality graphics, and convenient programmatic interfaces. It has become the de facto standard for robotics simulation in the ROS ecosystem, enabling researchers and developers to test algorithms, design robots, and conduct experiments in a safe, controlled environment.

Gazebo operates on a client-server architecture where the server handles physics simulation, rendering, and sensor simulation, while clients provide user interfaces and programmatic access. This architecture allows for both interactive experimentation and automated testing.

### Physics Simulation Engine

Gazebo uses the Open Dynamics Engine (ODE) as its primary physics engine, though it also supports Bullet and Simbody. The physics engine handles:
- Collision detection and response
- Rigid body dynamics
- Joint constraints and limits
- Force and torque application

The simulation operates in discrete time steps, typically at 1000 Hz, ensuring accurate and stable physics calculations. This high frequency is crucial for maintaining numerical stability, especially when dealing with complex multi-body systems.

### Sensor Simulation

One of Gazebo's key strengths is its comprehensive sensor simulation capabilities. It supports various sensor types:
- Camera sensors (monocular, stereo, depth)
- LIDAR sensors (2D and 3D)
- IMU sensors
- Force/torque sensors
- GPS and magnetometer sensors
- Contact sensors

These sensors generate realistic data that closely matches their real-world counterparts, making simulation-to-reality transfer more feasible.

## Gazebo Setup and Configuration

### Installation and Dependencies

To install Gazebo with ROS2 support, you'll need to install both the Gazebo simulator and the ROS2 Gazebo packages:

```bash
# Install Gazebo Fortress (or Garden/Harmonic depending on your ROS2 version)
sudo apt-get install gazebo11 libgazebo11-dev

# Install ROS2 Gazebo packages
sudo apt-get install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-plugins
```

### Basic Gazebo Launch

Here's a simple Python script to launch Gazebo with a basic world:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import subprocess
import time

class GazeboLauncher(Node):
    def __init__(self):
        super().__init__('gazebo_launcher')
        self.get_logger().info('Gazebo Launcher Node Started')

    def launch_gazebo(self):
        """Launch Gazebo with a basic world"""
        try:
            # Launch Gazebo with empty world
            subprocess.Popen(['gzserver', '--verbose', 'empty.world'])
            time.sleep(2)  # Wait for server to start

            # Launch Gazebo client (GUI)
            subprocess.Popen(['gzclient', '--verbose'])

            self.get_logger().info('Gazebo launched successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to launch Gazebo: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    launcher = GazeboLauncher()

    # Launch Gazebo
    launcher.launch_gazebo()

    try:
        rclpy.spin(launcher)
    except KeyboardInterrupt:
        launcher.get_logger().info('Shutting down Gazebo Launcher')
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## URDF/SDF Robot Models

### Understanding URDF and SDF

URDF (Unified Robot Description Format) is an XML format used in ROS to describe robot models. It defines the kinematic and dynamic properties of a robot, including:
- Links (rigid bodies)
- Joints (connections between links)
- Visual and collision properties
- Inertial properties
- Materials and colors

SDF (Simulation Description Format) is Gazebo's native format that extends URDF capabilities with simulation-specific features like:
- Physics properties
- Sensors
- Plugins
- World descriptions

### Creating a Simple Robot Model

Here's an example URDF for a simple wheeled robot:

```xml
<?xml version="1.0"?>
<robot name="simple_robot">
  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.15"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.3 0.15"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Left Wheel -->
  <link name="left_wheel">
    <visual>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
      <origin rpy="1.57079632679 0 0"/>
      <material name="black">
        <color rgba="0 0 0 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
      <origin rpy="1.57079632679 0 0"/>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Right Wheel -->
  <link name="right_wheel">
    <visual>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
      <origin rpy="1.57079632679 0 0"/>
      <material name="black">
        <color rgba="0 0 0 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
      <origin rpy="1.57079632679 0 0"/>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Joints -->
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="0 0.2 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel"/>
    <origin xyz="0 -0.2 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>
</robot>
```

## Physics Simulation

### Understanding Physics Parameters

Physics simulation in Gazebo is controlled by several key parameters that affect the accuracy and stability of the simulation:

- **Real Time Update Rate**: How often the physics engine updates in real time (Hz)
- **Max Step Size**: Maximum time step for physics calculations (seconds)
- **Real Time Factor**: Desired speedup of simulation compared to real time

Here's an example physics configuration file:

```xml
<?xml version="1.0"?>
<sdf version="1.7">
  <world name="default">
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>
  </world>
</sdf>
```

### Implementing Physics-Based Control

Here's a Python example that demonstrates physics-based control in Gazebo:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import math

class GazeboController(Node):
    def __init__(self):
        super().__init__('gazebo_controller')

        # Publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscriber for laser scan data
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Timer for control loop
        self.timer = self.create_timer(0.1, self.control_loop)

        self.scan_data = None
        self.get_logger().info('Gazebo Controller Node Started')

    def scan_callback(self, msg):
        """Callback for laser scan data"""
        self.scan_data = msg

    def control_loop(self):
        """Main control loop"""
        if self.scan_data is None:
            return

        cmd_vel = Twist()

        # Simple obstacle avoidance algorithm
        min_distance = min(self.scan_data.ranges)

        if min_distance < 1.0:  # If obstacle is closer than 1 meter
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.5  # Turn right
        else:
            cmd_vel.linear.x = 0.5  # Move forward
            cmd_vel.angular.z = 0.0

        self.cmd_vel_pub.publish(cmd_vel)

def main(args=None):
    rclpy.init(args=args)
    controller = GazeboController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down Gazebo Controller')
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Environment Building

### Creating Custom Worlds

Gazebo worlds are defined in SDF format and can include:
- Models and their initial positions
- Physics parameters
- Lighting and environment settings
- Plugins and sensors

Here's an example world file with obstacles:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="maze_world">
    <!-- Include ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Include sun -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Create walls -->
    <model name="wall_1">
      <pose>0 5 0 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>10 0.2 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 0.2 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Add more walls to create a maze -->
    <model name="wall_2">
      <pose>5 0 0 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.2 10 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.2 10 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Add some objects for navigation -->
    <model name="box_1">
      <pose>-2 -2 0 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
          <material>
            <ambient>1 0 0 1</ambient>
            <diffuse>1 0 0 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

## Practical Exercises

### Exercise 1: Simple Robot Simulation

Create a simple differential drive robot in Gazebo and implement a basic navigation algorithm.

**Requirements:**
1. Create a URDF model of a wheeled robot
2. Launch the robot in Gazebo
3. Implement a ROS2 node that controls the robot to move in a square pattern
4. Add a laser scanner to the robot
5. Implement obstacle avoidance behavior

**Steps:**
1. Create the robot URDF file
2. Create a launch file to spawn the robot in Gazebo
3. Write a ROS2 node for square pattern navigation
4. Add a laser scanner to the robot model
5. Implement obstacle detection and avoidance

### Exercise 2: Multi-Robot Simulation

Set up a simulation with multiple robots that interact with each other.

**Requirements:**
1. Spawn two identical robots in the same world
2. Implement communication between robots
3. Create a leader-follower behavior
4. Add collision avoidance between robots

## Chapter Summary

This chapter covered the fundamentals of Gazebo simulation for robotics development:

1. **Gazebo Architecture**: Understanding the client-server architecture and physics simulation engine
2. **Model Creation**: Creating URDF/SDF models for robots and environments
3. **Physics Simulation**: Configuring physics parameters and implementing physics-based control
4. **Environment Building**: Creating custom worlds and simulation scenarios
5. **ROS2 Integration**: Connecting Gazebo with ROS2 for robot control and sensing

Gazebo provides a powerful platform for robotics development, allowing for safe, repeatable, and cost-effective testing of algorithms and robot designs. The combination of realistic physics simulation, comprehensive sensor modeling, and tight ROS2 integration makes it an essential tool for robotics research and development.

## Further Reading

- Gazebo Classic Documentation: http://gazebosim.org/tutorials
- ROS2 with Gazebo: http://classic.gazebosim.org/tutorials?tut=ros2_overview
- URDF Tutorials: http://wiki.ros.org/urdf/Tutorials
- Physics Simulation in Robotics: "Robotics, Vision and Control" by Peter Corke

## Assessment Questions

1. Explain the difference between URDF and SDF formats in robotics simulation.
2. What are the key physics parameters that affect simulation stability?
3. Describe the process of spawning a robot model in Gazebo using ROS2.
4. How would you implement a collision avoidance algorithm using laser scan data in Gazebo?
5. What are the advantages and limitations of simulation compared to real-world testing?

import TranslateButton from '@site/src/components/TranslateButton';

<TranslateButton content={`# چیپٹر 4: گزیبو سیمولیشن کی بنیاد

## سیکھنے کے مقاصد

اس چیپٹر کے اختتام تک آپ کو درج ذیل کرنا ممکن ہوگا:
- گزیبو سیمولیشن ماحول کی بنیادی باتوں کو سمجھنا
- روبوٹکس پروجیکٹس کے لیے گزیبو سیٹ اپ اور کنفیگر کرنا
- گزیبو میں روبوٹ ماڈلز بنانا اور ان میں تبدیلیاں کرنا
- بنیادی سیمولیشن کے منظر نامے اور تجربات نافذ کرنا
- روبوٹ کنٹرول کے لیے گزیبو کو ROS2 کے ساتھ ضم کرنا

## نظریاتی بنیادیں

### گزیبو کا تعارف

گزیبو ایک طاقتور اوپن سورس روبوٹکس سیمولیٹر ہے جو حقیقی فزکس سیمولیشن، اعلیٰ معیار کے گریفکس، اور موزوں پروگرامنگ انٹرفیس فراہم کرتا ہے۔ یہ ROS ماحول کے لیے روبوٹکس سیمولیشن کا ایک معیاری معیار بن گیا ہے، جو تحقیق کاروں اور ڈیولپرز کو الگورتھم کی جانچ، روبوٹس کی ڈیزائن، اور تجربات کرنے کے قابل محفوظ، کنٹرول شدہ ماحول فراہم کرتا ہے۔

گزیبو ایک کلائنٹ-سرور آرکیٹیکچر پر کام کرتا ہے جہاں سرور فزکس سیمولیشن، رینڈرنگ، اور سینسر سیمولیشن کو ہینڈل کرتا ہے، جبکہ کلائنٹس صارف انٹرفیسز اور پروگرامنگ تک رسائی فراہم کرتے ہیں۔ یہ آرکیٹیکچر دونوں تعاملی تجربہ اور خودکار جانچ کو فروغ دیتا ہے۔

### فزکس سیمولیشن انجن

گزیبو Open Dynamics Engine (ODE) کو اس کا بنیادی فزکس انجن کے طور پر استعمال کرتا ہے، اگرچہ یہ Bullet اور Simbody کو بھی سپورٹ کرتا ہے۔ فزکس انجن یہ کام انجام دیتا ہے:
- کولیژن ڈیٹیکشن اور ریسپانس
- ریجڈ باڈی ڈائنیمکس
- جوائنٹ کنٹرولز اور حدود
- فورس اور ٹورک ایپلیکیشن

سیمولیشن ڈسکریٹ ٹائم اسٹیپس پر کام کرتا ہے، عام طور پر 1000 ہرٹز پر، جو تیز اور مستحکم فزکس کیلکولیشن کو یقینی بناتا ہے۔ یہ اعلیٰ فریکوینسی عددی استحکام کو برقرار رکھنے کے لیے انتہائی ضروری ہے، خاص طور پر جب جامع ملٹی-بodies سسٹمز کے ساتھ کام کیا جارہا ہو۔

### سینسر سیمولیشن

گزیبو کی ایک اہم طاقت اس کی جامع سینسر سیمولیشن کی صلاحیات ہے۔ یہ متعدد سینسر کی اقسام کو سپورٹ کرتا ہے:
- کیمرہ سینسرز (مونوکولر، سٹیریو، ڈیپتھ)
- لیڈار سینسرز (2D اور 3D)
- IMU سینسرز
- فورس/ٹورک سینسرز
- GPS اور میگنیٹومیٹر سینسرز
- کنٹیکٹ سینسرز

یہ سینسر حقیقی دنیا کے مطابق حقیقی ڈیٹا جنریٹ کرتے ہیں، جو سیمولیشن سے حقیقت میں ٹرانسفر کو زیادہ قابل عمل بناتا ہے۔

## گزیبو سیٹ اپ اور کنفیگریشن

### انسٹالیشن اور انحصار

ROS2 سپورٹ کے ساتھ گزیبو انسٹال کرنے کے لیے، آپ کو گزیبو سیمولیٹر اور ROS2 گزیبو پیکجز دونوں کو انسٹال کرنا ہوگا:

\`\`\`bash
# گزیبو فورٹریس (یا گارڈن/ہارمونک) انسٹال کریں (آپ کے ROS2 ورژن پر منحصر ہے)
sudo apt-get install gazebo11 libgazebo11-dev

# ROS2 گزیبو پیکجز انسٹال کریں
sudo apt-get install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-plugins
\`\`\`

### بنیادی گزیبو لانچ

یہاں گزیبو کو بنیادی دنیا کے ساتھ لانچ کرنے کے لیے ایک سادہ پائی تھن اسکرپٹ ہے:

\`\`\`python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import subprocess
import time

class GazeboLauncher(Node):
    def __init__(self):
        super().__init__('gazebo_launcher')
        self.get_logger().info('Gazebo Launcher Node Started')

    def launch_gazebo(self):
        """Launch Gazebo with a basic world"""
        try:
            # Launch Gazebo with empty world
            subprocess.Popen(['gzserver', '--verbose', 'empty.world'])
            time.sleep(2)  # Wait for server to start

            # Launch Gazebo client (GUI)
            subprocess.Popen(['gzclient', '--verbose'])

            self.get_logger().info('Gazebo launched successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to launch Gazebo: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    launcher = GazeboLauncher()

    # Launch Gazebo
    launcher.launch_gazebo()

    try:
        rclpy.spin(launcher)
    except KeyboardInterrupt:
        launcher.get_logger().info('Shutting down Gazebo Launcher')
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
\`\`\`

## URDF/SDF روبوٹ ماڈلز

### URDF اور SDF کو سمجھنا

URDF (متحدہ روبوٹ کی تفصیل کی شکل) ایک XML شکل ہے جو ROS میں روبوٹ ماڈلز کی تفصیل کے لیے استعمال ہوتی ہے۔ یہ روبوٹ کی کنیمیٹک اور ڈائنیمک خصوصیات کو بیان کرتا ہے، بشمول:
- لنکس (ریجڈ باڈیز)
- جوائنٹس (لنکس کے درمیان کنکشنز)
- وژوئل اور کولیژن خصوصیات
- انیشیل خصوصیات
- میٹریلز اور کلرز

SDF (سیمولیشن ڈسکرپشن فارمیٹ) گزیبو کی اصل شکل ہے جو URDF کی صلاحیات کو سیمولیشن مخصوص خصوصیات کے ساتھ وسعت دیتی ہے جیسے:
- فزکس خصوصیات
- سینسرز
- پلگ انز
- دنیا کی تفصیلات

### ایک سادہ روبوٹ ماڈل بنانا

یہاں ایک سادہ چکر والے روبوٹ کے لیے URDF مثال ہے:

\`\`\`xml
<?xml version="1.0"?>
<robot name="simple_robot">
  <!-- بیس لنک -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.15"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.3 0.15"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <!-- بائیں چکر -->
  <link name="left_wheel">
    <visual>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
      <origin rpy="1.57079632679 0 0"/>
      <material name="black">
        <color rgba="0 0 0 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
      <origin rpy="1.57079632679 0 0"/>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <!-- دائیں چکر -->
  <link name="right_wheel">
    <visual>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
      <origin rpy="1.57079632679 0 0"/>
      <material name="black">
        <color rgba="0 0 0 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
      <origin rpy="1.57079632679 0 0"/>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <!-- جوائنٹس -->
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="0 0.2 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel"/>
    <origin xyz="0 -0.2 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>
</robot>
\`\`\`

## فزکس سیمولیشن

### فزکس پیرامیٹرز کو سمجھنا

گزیبو میں فزکس سیمولیشن کو کئی کلیدی پیرامیٹرز کنٹرول کرتے ہیں جو سیمولیشن کی درستگی اور استحکام کو متاثر کرتے ہیں:

- **ریئل ٹائم اپ ڈیٹ ریٹ**: حقیقی وقت میں فزکس انجن کتنی بار اپ ڈیٹ ہوتا ہے (Hz)
- **زیادہ سے زیادہ اسٹیپ سائز**: فزکس کیلکولیشنز کے لیے زیادہ سے زیادہ ٹائم اسٹیپ (سیکنڈز)
- **ریئل ٹائم فیکٹر**: حقیقی وقت کے مقابلے میں سیمولیشن کی مطلوبہ رفتار

یہاں ایک مثال فزکس کنفیگریشن فائل ہے:

\`\`\`xml
<?xml version="1.0"?>
<sdf version="1.7">
  <world name="default">
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>
  </world>
\`\`\`

### فزکس-بیسڈ کنٹرول نافذ کرنا

یہاں گزیبو میں فزکس-بیسڈ کنٹرول کا ایک پائی تھن مثال ہے:

\`\`\`python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import math

class GazeboController(Node):
    def __init__(self):
        super().__init__('gazebo_controller')

        # رفتار کے کمانڈز کے لیے پبلشر
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # لیزر اسکین ڈیٹا کے لیے سبسکرائیب
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # کنٹرول لوپ کے لیے ٹائمر
        self.timer = self.create_timer(0.1, self.control_loop)

        self.scan_data = None
        self.get_logger().info('Gazebo Controller Node Started')

    def scan_callback(self, msg):
        """لیزر اسکین ڈیٹا کے لیے کال بیک"""
        self.scan_data = msg

    def control_loop(self):
        """مرکزی کنٹرول لوپ"""
        if self.scan_data is None:
            return

        cmd_vel = Twist()

        # سادہ رکاوٹ سے بچاؤ الگورتھم
        min_distance = min(self.scan_data.ranges)

        if min_distance < 1.0:  # اگر رکاوٹ 1 میٹر سے کم کے فاصلے پر ہے
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.5  # دائیں مڑیں
        else:
            cmd_vel.linear.x = 0.5  # آگے بڑھیں
            cmd_vel.angular.z = 0.0

        self.cmd_vel_pub.publish(cmd_vel)

def main(args=None):
    rclpy.init(args=args)
    controller = GazeboController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down Gazebo Controller')
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
\`\`\`

## ماحول کی تعمیر

### کسٹم دنیاؤں کو تخلیق کرنا

گزیبو کی دنیا SDF شکل میں وضاحت کی جاتی ہے اور اس میں شامل ہوسکتا ہے:
- ماڈلز اور ان کی ابتدائی پوزیشنز
- فزکس پیرامیٹرز
- لائٹنگ اور ماحول کی ترتیبات
- پلگ انز اور سینسرز

یہاں رکاوٹوں کے ساتھ ایک مثال دنیا فائل ہے:

\`\`\`xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="maze_world">
    <!-- زمینی سطح شامل کریں -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- سورج شامل کریں -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- دیواریں بنائیں -->
    <model name="wall_1">
      <pose>0 5 0 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>10 0.2 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 0.2 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- زیادہ دیواریں شامل کریں تاکہ ایک میز بن سکے -->
    <model name="wall_2">
      <pose>5 0 0 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.2 10 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.2 10 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- نیوی گیشن کے لیے کچھ اشیاء شامل کریں -->
    <model name="box_1">
      <pose>-2 -2 0 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
          <material>
            <ambient>1 0 0 1</ambient>
            <diffuse>1 0 0 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
  </world>
\`\`\`

## عملی مشقیں

### مشق 1: سادہ روبوٹ سیمولیشن

گزیبو میں ایک سادہ ڈفیشل ڈرائیو روبوٹ تخلیق کریں اور بنیادی نیوی گیشن الگورتھم نافذ کریں۔

**ضرورتیں:**
1. ایک URDF ماڈل کو ایک چکر والے روبوٹ کے لیے بنائیں
2. گزیبو میں روبوٹ کو لانچ کریں
3. ایک ROS2 نوڈ کو نافذ کریں جو روبوٹ کو مربع نمونہ میں منتقل کرے
4. روبوٹ میں ایک لیزر اسکینر شامل کریں
5. رکاوٹ سے بچاؤ کا رویہ نافذ کریں

**قدم:**
1. روبوٹ URDF فائل بنائیں
2. گزیبو میں روبوٹ کو اسپون کرنے کے لیے ایک لانچ فائل بنائیں
3. مربع نمونہ نیوی گیشن کے لیے ایک ROS2 نوڈ لکھیں
4. روبوٹ ماڈل میں ایک لیزر اسکینر شامل کریں
5. رکاوٹ کا پتہ لگانے اور بچاؤ نافذ کریں

### مشق 2: ملٹی-روبوٹ سیمولیشن

ایک ہی دنیا میں متعدد روبوٹس کے ساتھ ایک سیمولیشن سیٹ اپ کریں جو ایک دوسرے کے ساتھ تعامل کریں۔

**ضرورتیں:**
1. ایک ہی دنیا میں دو یکساں روبوٹس اسپون کریں
2. روبوٹس کے درمیان رابطہ نافذ کریں
3. ایک لیڈر-فالوور رویہ بنائیں
4. روبوٹس کے درمیان کولیژن ایوائڈینس شامل کریں

## چیپٹر کا خلاصہ

اس چیپٹر نے روبوٹکس ڈویلپمنٹ کے لیے گزیبو سیمولیشن کی بنیادی باتوں کو کور کیا:

1. **گزیبو آرکیٹیکچر**: کلائنٹ-سرور آرکیٹیکچر اور فزکس سیمولیشن انجن کو سمجھنا
2. **ماڈل تخلیق**: روبوٹس اور ماحول کے لیے URDF/SDF ماڈلز بنانا
3. **فزکس سیمولیشن**: فزکس پیرامیٹرز کنفیگر کرنا اور فزکس-بیسڈ کنٹرول نافذ کرنا
4. **ماحول کی تعمیر**: کسٹم دنیاؤں اور سیمولیشن منظر ناموں کو تخلیق کرنا
5. **ROS2 انضمام**: روبوٹ کنٹرول اور سینسنگ کے لیے گزیبو کو ROS2 کے ساتھ ضم کرنا

گزیبو روبوٹکس ڈویلپمنٹ کے لیے ایک طاقتور پلیٹ فارم فراہم کرتا ہے، جو الگورتھم اور روبوٹ ڈیزائنز کی محفوظ، دہرائی والی، اور قیمت کے لحاظ سے مؤثر جانچ کے قابل بناتا ہے۔ حقیقی فزکس سیمولیشن، جامع سینسر ماڈلنگ، اور ROS2 کے ساتھ تنگ انضمام کا مجموعہ اسے روبوٹکس کی تحقیق اور ڈویلپمنٹ کے لیے ایک ضروری ٹول بناتا ہے۔

## مزید پڑھنا

- گزیبو کلاسک ڈاکومنٹیشن: http://gazebosim.org/tutorials
- ROS2 کے ساتھ گزیبو: http://classic.gazebosim.org/tutorials?tut=ros2_overview
- URDF ٹوٹوریلز: http://wiki.ros.org/urdf/Tutorials
- روبوٹکس میں فزکس سیمولیشن: "Robotics, Vision and Control" by Peter Corke

## جائزہ سوالات

1. روبوٹکس سیمولیشن میں URDF اور SDF فارمیٹس کے درمیان فرق کی وضاحت کریں۔
2. وہ کون سے کلیدی فزکس پیرامیٹرز ہیں جو سیمولیشن کے استحکام کو متاثر کرتے ہیں؟
3. گزیبو میں روبوٹ ماڈل کو ROS2 کے ساتھ اسپون کرنے کا عمل بیان کریں۔
4. گزیبو میں لیزر اسکین ڈیٹا کا استعمال کرتے ہوئے کولیژن ایوائڈینس الگورتھم کیسے نافذ کریں گے؟
5. حقیقی دنیا کی جانچ کے مقابلے میں سیمولیشن کے کیا فوائد اور حدود ہیں؟
`} />

`} />