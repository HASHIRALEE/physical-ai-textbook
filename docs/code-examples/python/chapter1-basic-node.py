#!/usr/bin/env python3
# Basic ROS2 node for Physical AI system
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class PhysicalAIBasicNode(Node):
    def __init__(self):
        super().__init__('physical_ai_basic_node')
        self.publisher_ = self.create_publisher(String, 'physical_ai_topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Physical AI message: {self.i}' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    physical_ai_basic_node = PhysicalAIBasicNode()
    rclpy.spin(physical_ai_basic_node)
    physical_ai_basic_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()