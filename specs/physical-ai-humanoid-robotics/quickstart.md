# Quickstart Guide: Physical AI and Humanoid Robotics Textbook

## Prerequisites

Before starting development on the Physical AI and Humanoid Robotics textbook, ensure you have the following installed:

- **Node.js**: Version 18.x or higher
- **npm**: Version 8.x or higher (usually comes with Node.js)
- **Python**: Version 3.8 or higher
- **ROS2**: Humble Hawksbill distribution
- **Git**: Version control system

## Environment Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Install Node.js Dependencies
```bash
npm install
```

### 3. Setup Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### 4. ROS2 Environment Setup
```bash
# Source ROS2 Humble
source /opt/ros/humble/setup.bash

# If using custom workspace
source ~/ros2_ws/install/setup.bash
```

## Project Structure

```
physical-ai-textbook/
├── docs/                    # Docusaurus content
│   ├── intro.md
│   ├── chapters/            # Textbook chapters
│   ├── tutorials/           # Hands-on tutorials
│   ├── code-examples/       # Python/ROS2 code examples
│   └── exercises/           # Chapter exercises
├── src/                     # Custom React components
├── static/                  # Static assets (images, models)
├── docusaurus.config.js     # Docusaurus configuration
├── package.json             # Node.js dependencies
├── requirements.txt         # Python dependencies
└── README.md
```

## Running the Development Server

### 1. Start Docusaurus Development Server
```bash
npm start
```
This will start the development server at `http://localhost:3000`

### 2. Watch for Changes
The development server automatically reloads when you make changes to:
- Markdown files in `docs/`
- Configuration files
- Custom components in `src/`

## Creating a New Chapter

### 1. Use the Chapter Template
Create a new markdown file in `docs/chapters/` using this structure:

```markdown
---
id: ch01-introduction-to-physical-ai
title: Introduction to Physical AI
sidebar_label: Chapter 1: Introduction
sidebar_position: 1
description: Introduction to the fundamentals of Physical AI and embodied intelligence
---

## Learning Objectives

- Understand the fundamental concepts of Physical AI
- Distinguish between traditional AI and Physical AI
- Recognize the importance of embodiment in intelligent systems

## Core Concepts

[Theoretical foundations of Physical AI]

## Hands-on Tutorial

[Step-by-step practical tutorial]

## Code Implementation

```python
# Python code example
print("Hello Physical AI")
```

## Exercises & Challenges

1. [Exercise description]
2. [Exercise description]

## Further Reading

- [Reference 1]
- [Reference 2]

## Chapter Summary

[Key takeaways from the chapter]
```

### 2. Add Chapter to Sidebar
Update the `_category_.json` file in `docs/chapters/` to include your new chapter:

```json
{
  "label": "Chapters",
  "position": 2,
  "link": {
    "type": "generated-index",
    "description": "Learn about Physical AI and Humanoid Robotics"
  }
}
```

## Adding Code Examples

### 1. Python Examples
Place Python code in `docs/code-examples/python/`:

```python
# docs/code-examples/python/kinematics/simple_arm.py
import numpy as np

def forward_kinematics(joint_angles):
    """Calculate end-effector position from joint angles"""
    # Implementation here
    pass
```

### 2. ROS2 Examples
Place ROS2 code in `docs/code-examples/ros2/`:

```
docs/code-examples/ros2/
├── launch/
├── config/
├── scripts/
└── nodes/
```

## Working with ROS2 Components

### 1. Creating a New ROS2 Node
```python
# Example ROS2 Python node
import rclpy
from rclpy.node import Node

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')
        # Node implementation
```

### 2. Running ROS2 Examples
```bash
# Source ROS2
source /opt/ros/humble/setup.bash

# Run the node
ros2 run package_name node_name
```

## Simulation Environment

### 1. Setting up Gazebo
```bash
# Install Gazebo Harmonic
sudo apt install ros-humble-gazebo-*

# Launch simulation
ros2 launch package_name simulation.launch.py
```

### 2. Testing with Simulation
```bash
# Example: Launch robot in simulation
ros2 launch my_robot_gazebo my_robot_world.launch.py
```

## Building for Production

### 1. Build Static Site
```bash
npm run build
```

### 2. Serve Built Site Locally
```bash
npm run serve
```

## Running Tests

### 1. Linting and Formatting
```bash
# Check code style
npm run lint

# Format code
npm run format
```

### 2. Validate Links
```bash
npm run check-links
```

## Deployment

### GitHub Pages Deployment
The site is automatically deployed via GitHub Actions when changes are pushed to the main branch.

### Manual Deployment
```bash
GIT_USER=<Your GitHub Username> \
  CURRENT_BRANCH=main \
  USE_SSH=true \
  npm run deploy
```

## Troubleshooting

### Common Issues

1. **Port already in use**: Change the port with `npm start -- --port 3001`
2. **Missing Python dependencies**: Run `pip install -r requirements.txt`
3. **ROS2 not found**: Ensure ROS2 environment is sourced
4. **Build fails**: Try clearing cache with `npm run clear`

### Getting Help

- Check the [Docusaurus documentation](https://docusaurus.io)
- Review ROS2 Humble documentation
- Consult the project's issue tracker