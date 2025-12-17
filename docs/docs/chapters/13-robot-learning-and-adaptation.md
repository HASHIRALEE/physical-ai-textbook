---
sidebar_position: 13
title: "Chapter 13: Robot Learning and Adaptation"
---

# Chapter 13: Robot Learning and Adaptation

## Learning Objectives

By the end of this chapter, students will be able to:
- Understand fundamental machine learning concepts and their application to robotics
- Implement supervised, unsupervised, and reinforcement learning algorithms for robotic tasks
- Apply deep learning techniques to perception and control problems in robotics
- Design adaptive control systems that learn from experience
- Evaluate the performance of learning algorithms in robotic applications
- Address challenges specific to learning in robotic systems
- Implement transfer learning techniques for robotic skill acquisition
- Design safe and robust learning systems for real-world robotic deployment

## Theoretical Foundations

### Introduction to Machine Learning in Robotics

Machine learning has become integral to modern robotics, enabling robots to learn from experience, adapt to new situations, and improve their performance over time. Unlike traditional programming approaches where robots follow predetermined algorithms, machine learning allows robots to discover patterns, make decisions, and optimize their behavior based on data.

The application of machine learning to robotics presents unique challenges and opportunities:

**Perception Learning**: Robots must learn to interpret sensor data, recognize objects, understand scenes, and navigate environments. This includes computer vision, speech recognition, and sensor fusion tasks.

**Control Learning**: Robots can learn optimal control strategies, motor skills, and adaptive behaviors that allow them to perform tasks more effectively in varying conditions.

**Planning and Decision Making**: Learning algorithms can help robots develop better planning strategies, predict outcomes, and make decisions under uncertainty.

**Social Learning**: In human-robot interaction scenarios, robots can learn to understand human intentions, preferences, and social cues.

### Types of Learning in Robotics

**Supervised Learning**: In this approach, robots learn from labeled training data. Common applications include object recognition, where the robot learns to classify objects based on labeled images, or trajectory learning, where the robot learns to reproduce demonstrated movements.

**Unsupervised Learning**: Robots discover patterns and structure in unlabeled data. This is useful for clustering similar behaviors, identifying anomalies in sensor data, or learning representations of the environment.

**Reinforcement Learning**: Robots learn through trial and error, receiving rewards or penalties based on their actions. This approach is particularly powerful for control and decision-making tasks where the optimal behavior is not known in advance.

**Imitation Learning**: Robots learn by observing and mimicking human demonstrations. This approach is valuable for teaching complex manipulation tasks that are difficult to program manually.

### Learning Theory and Generalization

For robotic learning systems to be effective, they must generalize from training experiences to new situations. Key concepts include:

**Bias-Variance Tradeoff**: The balance between model complexity and generalization ability. Simple models may underfit (high bias), while complex models may overfit (high variance) to training data.

**Sample Complexity**: The amount of data required for effective learning, which is particularly important in robotics where data collection can be expensive and time-consuming.

**Transfer Learning**: The ability to apply knowledge learned in one context to new but related tasks, which is crucial for efficient learning in robotics.

## Supervised Learning for Robotics

### Classification and Regression

```python
#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Any
import pickle

class RobotPerceptionLearner:
    def __init__(self):
        """Initialize robot perception learning system"""
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_names = []
        self.target_names = []

    def prepare_sensor_data(self, sensor_readings: np.ndarray,
                           labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sensor data for learning"""
        # Normalize sensor readings
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(sensor_readings)

        # Encode labels if they're categorical
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)

        self.scalers['sensor'] = scaler
        self.label_encoders['target'] = label_encoder

        return normalized_data, encoded_labels

    def train_object_classifier(self, X: np.ndarray, y: np.ndarray,
                               model_type: str = 'random_forest') -> Dict:
        """Train object classification model"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Initialize model
        if model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'svm':
            model = SVC(kernel='rbf', random_state=42)
        elif model_type == 'logistic':
            model = LogisticRegression(random_state=42, max_iter=1000)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Train model
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        results = {
            'model': model,
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred),
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }

        self.models['object_classifier'] = results
        return results

    def train_pose_regressor(self, X: np.ndarray, y: np.ndarray,
                            model_type: str = 'random_forest') -> Dict:
        """Train pose estimation regression model"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Initialize model
        if model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'svm':
            model = SVR(kernel='rbf')
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Train model
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        results = {
            'model': model,
            'mse': mse,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }

        self.models['pose_regressor'] = results
        return results

    def predict_object(self, sensor_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict object class from sensor data"""
        if 'object_classifier' not in self.models:
            raise ValueError("Object classifier not trained")

        model = self.models['object_classifier']['model']
        scaler = self.scalers['sensor']
        label_encoder = self.label_encoders['target']

        # Normalize input
        normalized_data = scaler.transform(sensor_data.reshape(1, -1))

        # Predict
        prediction = model.predict(normalized_data)
        probabilities = model.predict_proba(normalized_data) if hasattr(model, 'predict_proba') else None

        # Decode label
        predicted_class = label_encoder.inverse_transform(prediction)[0]

        return predicted_class, probabilities

    def predict_pose(self, sensor_data: np.ndarray) -> np.ndarray:
        """Predict pose from sensor data"""
        if 'pose_regressor' not in self.models:
            raise ValueError("Pose regressor not trained")

        model = self.models['pose_regressor']['model']
        scaler = self.scalers['sensor']

        # Normalize input
        normalized_data = scaler.transform(sensor_data.reshape(1, -1))

        # Predict
        prediction = model.predict(normalized_data)

        return prediction

class RobotSkillLearner:
    def __init__(self):
        """Initialize robot skill learning system"""
        self.skill_models = {}
        self.skill_scalers = {}
        self.skill_sequences = []

    def learn_trajectory(self, demonstrations: List[np.ndarray],
                        skill_name: str) -> Dict:
        """Learn a skill from demonstrations using Gaussian Mixture Models"""
        from sklearn.mixture import GaussianMixture
        import scipy.stats as stats

        # Concatenate all demonstrations
        all_demonstrations = np.vstack(demonstrations)

        # Fit Gaussian Mixture Model
        gmm = GaussianMixture(n_components=5, random_state=42)
        gmm.fit(all_demonstrations)

        # Store model
        self.skill_models[skill_name] = gmm

        # Calculate statistics for the skill
        skill_stats = {
            'mean_trajectory': np.mean(all_demonstrations, axis=0),
            'std_trajectory': np.std(all_demonstrations, axis=0),
            'n_demonstrations': len(demonstrations),
            'model': gmm
        }

        return skill_stats

    def generate_trajectory(self, skill_name: str, length: int = 100) -> np.ndarray:
        """Generate a trajectory for the learned skill"""
        if skill_name not in self.skill_models:
            raise ValueError(f"Skill {skill_name} not learned")

        gmm = self.skill_models[skill_name]

        # Generate samples from the GMM
        trajectory, _ = gmm.sample(n_samples=length)

        return trajectory

    def adapt_trajectory(self, skill_name: str, context: Dict) -> np.ndarray:
        """Adapt learned trajectory based on context"""
        if skill_name not in self.skill_models:
            raise ValueError(f"Skill {skill_name} not learned")

        # In a real implementation, this would adapt the trajectory based on context
        # For now, we'll just generate a new trajectory
        trajectory = self.generate_trajectory(skill_name)

        # Apply context-based adaptations
        if 'target_position' in context:
            target_pos = np.array(context['target_position'])
            # Adjust trajectory to reach target position
            trajectory[:, :3] = trajectory[:, :3] + (target_pos - trajectory[-1, :3])

        return trajectory

# Example usage
if __name__ == "__main__":
    # Create perception learner
    perception_learner = RobotPerceptionLearner()

    # Simulate sensor data (e.g., from cameras, LiDAR, IMU)
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    # Simulated sensor readings
    sensor_data = np.random.randn(n_samples, n_features)

    # Simulated object labels (0: cube, 1: sphere, 2: cylinder)
    object_labels = np.random.choice([0, 1, 2], size=n_samples)

    # Train object classifier
    print("Training object classifier...")
    classification_results = perception_learner.train_object_classifier(
        sensor_data, object_labels, model_type='random_forest'
    )
    print(f"Classification accuracy: {classification_results['accuracy']:.3f}")

    # Simulate pose data (x, y, z, roll, pitch, yaw)
    pose_data = np.random.randn(n_samples, 6)
    object_poses = np.random.randn(n_samples, 6)  # Target poses

    # Train pose regressor
    print("Training pose regressor...")
    pose_results = perception_learner.train_pose_regressor(
        sensor_data, object_poses, model_type='random_forest'
    )
    print(f"Pose regression MSE: {pose_results['mse']:.3f}")

    # Create skill learner
    skill_learner = RobotSkillLearner()

    # Simulate demonstrations (e.g., 5 demonstrations of a pick-and-place task)
    demonstrations = []
    for i in range(5):
        # Each demonstration is a sequence of joint positions over time
        demo = np.random.randn(50, 7)  # 50 time steps, 7 joints
        demonstrations.append(demo)

    # Learn the skill
    print("Learning pick-and-place skill...")
    skill_stats = skill_learner.learn_trajectory(demonstrations, "pick_place")
    print(f"Learned skill from {skill_stats['n_demonstrations']} demonstrations")

    # Generate a new trajectory
    new_trajectory = skill_learner.generate_trajectory("pick_place", length=50)
    print(f"Generated trajectory shape: {new_trajectory.shape}")
```

### Deep Learning for Perception

```python
#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt

class RobotVisionDataset(Dataset):
    """Dataset for robot vision tasks"""
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = torch.FloatTensor(images)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

class ConvolutionalFeatureExtractor(nn.Module):
    """CNN for extracting visual features from robot sensors"""
    def __init__(self, input_channels: int = 3, num_classes: int = 10):
        super(ConvolutionalFeatureExtractor, self).__init__()

        # Convolutional layers for feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Calculate the size of the flattened features
        self.feature_size = 128 * 4 * 4  # After 3 max pooling layers of 2x2

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Extract features through conv layers
        x = self.conv_layers(x)

        # Flatten for fully connected layers
        x = x.view(-1, self.feature_size)

        # Apply fully connected layers
        x = self.fc_layers(x)

        return x

class RobotActionNet(nn.Module):
    """Network for learning robot actions from sensor data"""
    def __init__(self, sensor_dim: int, action_dim: int, hidden_dim: int = 256):
        super(RobotActionNet, self).__init__()

        self.sensor_encoder = nn.Sequential(
            nn.Linear(sensor_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        self.action_decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, sensor_input):
        encoded = self.sensor_encoder(sensor_input)
        action = self.action_decoder(encoded)
        return action

class DeepVisionLearner:
    def __init__(self, device: str = 'cpu'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()

    def create_model(self, input_channels: int, num_classes: int):
        """Create CNN model for vision tasks"""
        self.model = ConvolutionalFeatureExtractor(input_channels, num_classes)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, train_loader: DataLoader, num_epochs: int = 10):
        """Train the vision model"""
        self.model.train()
        losses = []

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            losses.append(avg_loss)
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

        return losses

    def evaluate(self, test_loader: DataLoader):
        """Evaluate the model"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        return accuracy

class RobotLearningSystem:
    def __init__(self):
        """Initialize the robot learning system"""
        self.perception_learner = DeepVisionLearner()
        self.action_learner = RobotActionNet(sensor_dim=10, action_dim=6)
        self.memory_buffer = []
        self.experience_buffer = []

    def learn_from_experience(self, state: np.ndarray, action: np.ndarray,
                             reward: float, next_state: np.ndarray, done: bool):
        """Store experience for reinforcement learning"""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }
        self.experience_buffer.append(experience)

        # Keep only recent experiences (replay buffer)
        if len(self.experience_buffer) > 10000:
            self.experience_buffer.pop(0)

    def train_perception(self, images: np.ndarray, labels: np.ndarray):
        """Train perception system"""
        # Create dataset
        dataset = RobotVisionDataset(images, labels)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Create and train model
        self.perception_learner.create_model(input_channels=images.shape[1],
                                           num_classes=len(np.unique(labels)))
        losses = self.perception_learner.train(dataloader, num_epochs=5)

        return losses

# Example usage
if __name__ == "__main__":
    # Create learning system
    learning_system = RobotLearningSystem()

    # Simulate vision data (e.g., 1000 images of 64x64x3)
    images = np.random.randn(1000, 3, 64, 64).astype(np.float32)
    labels = np.random.randint(0, 10, 1000)  # 10 object classes

    print("Training perception system...")
    losses = learning_system.train_perception(images, labels)
    print(f"Final loss: {losses[-1]:.4f}")
```

## Reinforcement Learning for Robotics

### Q-Learning and Deep Q-Networks

```python
#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
from typing import Tuple, List, Dict, Any
import matplotlib.pyplot as plt

# Named tuple for experience replay
Experience = namedtuple('Experience',
                       ['state', 'action', 'reward', 'next_state', 'done'])

class QNetwork(nn.Module):
    """Deep Q-Network for learning state-action values"""
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        super(QNetwork, self).__init__()

        self.fc_layers = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, state):
        return self.fc_layers(state)

class ExperienceReplay:
    """Experience replay buffer for DQN"""
    def __init__(self, buffer_size: int = 10000):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        """Randomly sample experiences from buffer"""
        return random.sample(self.buffer, k=batch_size)

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size: int, action_size: int, lr: float = 0.001,
                 gamma: float = 0.99, epsilon: float = 1.0,
                 epsilon_decay: float = 0.995, epsilon_min: float = 0.01):
        """Initialize DQN agent"""
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Neural networks
        self.qnetwork_local = QNetwork(state_size, action_size)
        self.qnetwork_target = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        # Replay memory
        self.memory = ExperienceReplay(buffer_size=10000)

        # Update target network
        self.update_target_network()

    def step(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """Take a step in the environment"""
        # Add experience to memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn if there are enough samples
        if len(self.memory) > 64:
            experiences = self.memory.sample(64)
            self.learn(experiences)

    def act(self, state: np.ndarray, add_noise: bool = False) -> int:
        """Choose action using epsilon-greedy policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # Get Q-values from local network
        q_values = self.qnetwork_local(state_tensor)

        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            # Exploit: choose best action
            return q_values.argmax().item()
        else:
            # Explore: choose random action
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences: List[Experience]):
        """Learn from sampled experiences"""
        states = torch.FloatTensor([e.state for e in experiences])
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor([e.next_state for e in experiences])
        dones = torch.BoolTensor([e.done for e in experiences])

        # Get Q-values for current states
        current_q_values = self.qnetwork_local(states).gather(1, actions.unsqueeze(1))

        # Get max Q-values for next states from target network
        next_q_values = self.qnetwork_target(next_states).detach().max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        # Minimize loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        """Update target network with local network weights"""
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

class ContinuousDQNAgent:
    """DQN agent for continuous action spaces"""
    def __init__(self, state_size: int, action_size: int, action_low: float = -1.0,
                 action_high: float = 1.0, lr: float = 0.001, gamma: float = 0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.gamma = gamma

        # Actor and Critic networks
        self.actor_local = self._build_actor(state_size, action_size)
        self.actor_target = self._build_actor(state_size, action_size)
        self.critic_local = self._build_critic(state_size, action_size)
        self.critic_target = self._build_critic(state_size, action_size)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr)

        # Replay memory
        self.memory = ExperienceReplay(buffer_size=10000)

        # Update target networks
        self.update_target_networks()

    def _build_actor(self, state_size: int, action_size: int) -> nn.Module:
        """Build actor network"""
        return nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
            nn.Tanh()  # Output in [-1, 1] range
        )

    def _build_critic(self, state_size: int, action_size: int) -> nn.Module:
        """Build critic network"""
        return nn.Sequential(
            nn.Linear(state_size + action_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def act(self, state: np.ndarray) -> np.ndarray:
        """Get action from actor network"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor_local(state_tensor).cpu().data.numpy()[0]

        # Scale action to appropriate range
        scaled_action = self.action_low + (action + 1.0) * (self.action_high - self.action_low) / 2.0

        return scaled_action

    def update_target_networks(self):
        """Update target networks"""
        self.actor_target.load_state_dict(self.actor_local.state_dict())
        self.critic_target.load_state_dict(self.critic_local.state_dict())

class RobotRLAgent:
    def __init__(self, state_size: int, action_size: int):
        """Initialize robot reinforcement learning agent"""
        self.dqn_agent = DQNAgent(state_size, action_size)
        self.continuous_agent = ContinuousDQNAgent(state_size, action_size)
        self.training_mode = True
        self.episode_rewards = []

    def train_step(self, state: np.ndarray, action: int, reward: float,
                   next_state: np.ndarray, done: bool):
        """Perform one training step"""
        if self.training_mode:
            self.dqn_agent.step(state, action, reward, next_state, done)

    def select_action(self, state: np.ndarray) -> int:
        """Select action using the agent's policy"""
        return self.dqn_agent.act(state)

    def evaluate_policy(self, env, n_episodes: int = 10) -> List[float]:
        """Evaluate the current policy"""
        rewards = []

        for episode in range(n_episodes):
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                state = next_state
                total_reward += reward

            rewards.append(total_reward)

        return rewards

# Example environment simulation
class SimpleRobotEnv:
    """Simple robot environment for testing RL algorithms"""
    def __init__(self, state_size: int = 4, action_size: int = 2):
        self.state_size = state_size
        self.action_size = action_size
        self.state = None
        self.reset()

    def reset(self):
        """Reset environment to initial state"""
        self.state = np.random.randn(self.state_size)
        return self.state

    def step(self, action: int):
        """Take an action in the environment"""
        # Simple reward based on state values
        reward = np.sum(self.state) * 0.1

        # Update state based on action (simplified dynamics)
        action_effect = np.zeros(self.state_size)
        if action < len(action_effect):
            action_effect[action] = 1.0

        self.state = self.state + action_effect * 0.1 + np.random.randn(self.state_size) * 0.01

        # Simple termination condition
        done = np.abs(np.sum(self.state)) > 10
        info = {}

        return self.state, reward, done, info

# Example usage
if __name__ == "__main__":
    # Create environment and agent
    env = SimpleRobotEnv(state_size=4, action_size=4)
    agent = RobotRLAgent(state_size=4, action_size=4)

    print("Training DQN agent...")

    # Training loop
    n_episodes = 100
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step_count = 0

        while not done and step_count < 100:  # Limit steps per episode
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.train_step(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            step_count += 1

        agent.episode_rewards.append(total_reward)

        if episode % 20 == 0:
            avg_reward = np.mean(agent.episode_rewards[-20:]) if len(agent.episode_rewards) >= 20 else np.mean(agent.episode_rewards)
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}, Epsilon: {agent.dqn_agent.epsilon:.3f}")

    print("Training completed!")
    print(f"Final average reward: {np.mean(agent.episode_rewards[-10:]):.2f}")
```

### Policy Gradient Methods

```python
#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

class PolicyNetwork(nn.Module):
    """Policy network for policy gradient methods"""
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        super(PolicyNetwork, self).__init__()

        self.fc_layers = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, state):
        logits = self.fc_layers(state)
        return F.softmax(logits, dim=-1)

class ValueNetwork(nn.Module):
    """Value network for advantage estimation"""
    def __init__(self, state_size: int, hidden_size: int = 64):
        super(ValueNetwork, self).__init__()

        self.fc_layers = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state):
        return self.fc_layers(state)

class ActorCriticAgent:
    def __init__(self, state_size: int, action_size: int, lr: float = 0.001):
        """Initialize Actor-Critic agent"""
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr

        # Networks
        self.policy_network = PolicyNetwork(state_size, action_size)
        self.value_network = ValueNetwork(state_size)

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=lr)

    def get_action(self, state: torch.Tensor) -> Tuple[int, float]:
        """Get action and log probability"""
        state = state.unsqueeze(0) if len(state.shape) == 1 else state

        # Get action probabilities from policy network
        action_probs = self.policy_network(state)
        dist = Categorical(action_probs)

        # Sample action
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.item()

    def update(self, states: List[torch.Tensor], actions: List[int],
               rewards: List[float], next_states: List[torch.Tensor],
               dones: List[bool]):
        """Update networks using Actor-Critic method"""
        states_tensor = torch.stack(states)
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)

        # Calculate value targets (rewards + gamma * V(next_state))
        with torch.no_grad():
            next_values = self.value_network(torch.stack(next_states)).squeeze()
            value_targets = rewards_tensor + 0.99 * next_values * (1 - torch.FloatTensor(dones))

        # Update value network
        current_values = self.value_network(states_tensor).squeeze()
        value_loss = F.mse_loss(current_values, value_targets)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Calculate advantages
        advantages = value_targets - current_values.detach()

        # Update policy network
        action_probs = self.policy_network(states_tensor)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions_tensor)

        policy_loss = -(log_probs * advantages).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

class ContinuousActorCritic(nn.Module):
    """Actor-Critic for continuous action spaces"""
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        super(ContinuousActorCritic, self).__init__()

        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # Actor head (outputs mean and std for Gaussian distribution)
        self.actor_mean = nn.Linear(hidden_size, action_size)
        self.actor_std = nn.Linear(hidden_size, action_size)

        # Critic head (outputs state value)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: return action mean, std, and state value"""
        shared_features = self.shared_layers(state)

        action_mean = torch.tanh(self.actor_mean(shared_features))
        action_std = F.softplus(self.actor_std(shared_features))
        state_value = self.critic(shared_features)

        return action_mean, action_std, state_value

class PPOAgent:
    """Proximal Policy Optimization agent"""
    def __init__(self, state_size: int, action_size: int, lr: float = 0.001,
                 clip_epsilon: float = 0.2, epochs: int = 4):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs

        # Actor-Critic network
        self.network = ContinuousActorCritic(state_size, action_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

    def get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action and log probability"""
        state = state.unsqueeze(0) if len(state.shape) == 1 else state

        action_mean, action_std, _ = self.network(state)
        dist = Normal(action_mean, action_std)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob

    def update(self, states: torch.Tensor, actions: torch.Tensor,
               old_log_probs: torch.Tensor, returns: torch.Tensor,
               advantages: torch.Tensor):
        """Update network using PPO objective"""
        for _ in range(self.epochs):
            action_mean, action_std, state_values = self.network(states)
            dist = Normal(action_mean, action_std)

            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)

            # PPO objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(state_values.squeeze(), returns)

            # Total loss
            total_loss = actor_loss + 0.5 * value_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

class RobotLearningEnvironment:
    def __init__(self):
        """Environment for robot learning"""
        self.agents = {}
        self.episode_history = []

    def register_agent(self, name: str, agent):
        """Register a learning agent"""
        self.agents[name] = agent

    def run_episode(self, agent_name: str, env, max_steps: int = 1000) -> Dict:
        """Run an episode with a specific agent"""
        agent = self.agents[agent_name]
        state = env.reset()
        total_reward = 0
        step_count = 0
        done = False

        episode_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
            'log_probs': []
        }

        while not done and step_count < max_steps:
            # Convert to tensor if needed
            if not isinstance(state, torch.Tensor):
                state_tensor = torch.FloatTensor(state)
            else:
                state_tensor = state

            if hasattr(agent, 'get_action'):
                action, log_prob = agent.get_action(state_tensor)

                # Store for learning
                episode_data['states'].append(state_tensor)
                episode_data['actions'].append(action)
                episode_data['log_probs'].append(log_prob)

                # Take action in environment
                next_state, reward, done, _ = env.step(action)

                # Store experience
                episode_data['rewards'].append(reward)
                episode_data['next_states'].append(torch.FloatTensor(next_state))
                episode_data['dones'].append(done)

                state = next_state
                total_reward += reward
                step_count += 1
            else:
                # For discrete action agents
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)

                # Store experience
                episode_data['states'].append(state)
                episode_data['actions'].append(action)
                episode_data['rewards'].append(reward)
                episode_data['next_states'].append(next_state)
                episode_data['dones'].append(done)

                state = next_state
                total_reward += reward
                step_count += 1

        episode_result = {
            'total_reward': total_reward,
            'steps': step_count,
            'data': episode_data
        }

        self.episode_history.append(episode_result)
        return episode_result

# Example usage
if __name__ == "__main__":
    # Create learning environment
    learning_env = RobotLearningEnvironment()

    # Create and register agents
    discrete_agent = RobotRLAgent(state_size=4, action_size=4)
    learning_env.register_agent('dqn_agent', discrete_agent)

    continuous_agent = PPOAgent(state_size=4, action_size=2)
    learning_env.register_agent('ppo_agent', continuous_agent)

    # Create environment
    env = SimpleRobotEnv(state_size=4, action_size=4)

    print("Running learning episodes...")

    # Run episodes with DQN agent
    for episode in range(50):
        result = learning_env.run_episode('dqn_agent', env)
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {result['total_reward']:.2f}")

    print("Learning completed!")
```

## Transfer Learning and Domain Adaptation

### Knowledge Transfer in Robotics

```python
#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Any
import copy

class FeatureExtractor(nn.Module):
    """Feature extractor that can be shared across tasks"""
    def __init__(self, input_size: int, feature_size: int = 128):
        super(FeatureExtractor, self).__init__()

        self.feature_layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, feature_size),
            nn.ReLU()
        )

    def forward(self, x):
        return self.feature_layers(x)

class TaskHead(nn.Module):
    """Task-specific head that can be trained independently"""
    def __init__(self, feature_size: int, output_size: int):
        super(TaskHead, self).__init__()

        self.task_layers = nn.Sequential(
            nn.Linear(feature_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.task_layers(x)

class MultiTaskRobotLearner:
    def __init__(self, input_size: int, feature_size: int = 128):
        """Initialize multi-task learning system"""
        self.feature_extractor = FeatureExtractor(input_size, feature_size)
        self.task_heads = nn.ModuleDict()
        self.optimizers = {}
        self.task_data = {}

    def add_task(self, task_name: str, output_size: int, lr: float = 0.001):
        """Add a new task to the system"""
        # Create task head
        task_head = TaskHead(self.feature_extractor.feature_layers[-2].out_features, output_size)
        self.task_heads[task_name] = task_head

        # Create optimizer for this task
        params = list(self.feature_extractor.parameters()) + list(task_head.parameters())
        self.optimizers[task_name] = optim.Adam(params, lr=lr)

        # Initialize data storage for this task
        self.task_data[task_name] = {'X': [], 'y': []}

    def train_task(self, task_name: str, X: torch.Tensor, y: torch.Tensor,
                   epochs: int = 10):
        """Train a specific task"""
        if task_name not in self.task_heads:
            raise ValueError(f"Task {task_name} not found")

        optimizer = self.optimizers[task_name]
        task_head = self.task_heads[task_name]

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Extract features
            features = self.feature_extractor(X)

            # Get task-specific output
            output = task_head(features)

            # Calculate loss
            loss = nn.MSELoss()(output, y)

            # Backpropagate
            loss.backward()
            optimizer.step()

    def transfer_to_new_task(self, source_task: str, target_task: str,
                           output_size: int, lr: float = 0.001):
        """Transfer knowledge from source task to target task"""
        # Add new task
        self.add_task(target_task, output_size, lr)

        # Copy feature extractor weights (frozen for transfer)
        self.task_heads[target_task].load_state_dict(
            self.task_heads[source_task].state_dict(), strict=False
        )

        # Freeze feature extractor for fine-tuning
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def predict(self, task_name: str, X: torch.Tensor) -> torch.Tensor:
        """Make prediction for a specific task"""
        if task_name not in self.task_heads:
            raise ValueError(f"Task {task_name} not found")

        features = self.feature_extractor(X)
        output = self.task_heads[task_name](features)

        return output

class DomainAdaptationLearner:
    def __init__(self, feature_size: int = 128):
        """Initialize domain adaptation system"""
        self.feature_extractor = FeatureExtractor(feature_size, feature_size)
        self.source_classifier = nn.Linear(feature_size, 10)  # 10 classes
        self.domain_classifier = nn.Linear(feature_size, 2)   # Source vs Target

        self.source_optimizer = optim.Adam(
            list(self.feature_extractor.parameters()) +
            list(self.source_classifier.parameters()), lr=0.001
        )

        self.domain_optimizer = optim.Adam(
            list(self.feature_extractor.parameters()) +
            list(self.domain_classifier.parameters()), lr=0.001
        )

    def train_source(self, source_data: Tuple[torch.Tensor, torch.Tensor],
                    epochs: int = 10):
        """Train on source domain data"""
        X, y = source_data

        for epoch in range(epochs):
            self.source_optimizer.zero_grad()

            features = self.feature_extractor(X)
            class_pred = self.source_classifier(features)

            class_loss = nn.CrossEntropyLoss()(class_pred, y)
            class_loss.backward()

            self.source_optimizer.step()

    def adapt_domain(self, source_data: Tuple[torch.Tensor, torch.Tensor],
                    target_data: Tuple[torch.Tensor, torch.Tensor], epochs: int = 10):
        """Adapt to target domain using domain adaptation"""
        source_X, source_y = source_data
        target_X, _ = target_data

        for epoch in range(epochs):
            # Train domain classifier
            self.domain_optimizer.zero_grad()

            # Source domain predictions
            source_features = self.feature_extractor(source_X)
            source_domain_pred = self.domain_classifier(source_features)
            source_domain_labels = torch.zeros(source_X.size(0), dtype=torch.long)

            # Target domain predictions
            target_features = self.feature_extractor(target_X)
            target_domain_pred = self.domain_classifier(target_features)
            target_domain_labels = torch.ones(target_X.size(0), dtype=torch.long)

            # Combine domain predictions
            all_domain_pred = torch.cat([source_domain_pred, target_domain_pred])
            all_domain_labels = torch.cat([source_domain_labels, target_domain_labels])

            domain_loss = nn.CrossEntropyLoss()(all_domain_pred, all_domain_labels)

            # Minimize domain loss (confuse the domain classifier)
            domain_loss.backward()

            self.domain_optimizer.step()

    def extract_features(self, X: torch.Tensor) -> torch.Tensor:
        """Extract domain-invariant features"""
        return self.feature_extractor(X)

class RobotSkillTransfer:
    def __init__(self):
        """Initialize robot skill transfer system"""
        self.skill_encoders = {}
        self.skill_decoders = {}
        self.transfer_models = {}

    def encode_skill(self, skill_name: str, demonstrations: List[np.ndarray]) -> torch.Tensor:
        """Encode a skill using an autoencoder"""
        if skill_name not in self.skill_encoders:
            # Create encoder-decoder pair
            encoder = nn.Sequential(
                nn.Linear(demonstrations[0].shape[-1], 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 16)  # Latent space
            )

            decoder = nn.Sequential(
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, demonstrations[0].shape[-1])
            )

            self.skill_encoders[skill_name] = encoder
            self.skill_decoders[skill_name] = decoder
            self.transfer_models[skill_name] = {
                'encoder': encoder,
                'decoder': decoder,
                'optimizer': optim.Adam(
                    list(encoder.parameters()) + list(decoder.parameters()),
                    lr=0.001
                )
            }

        # Train the autoencoder
        model = self.transfer_models[skill_name]
        optimizer = model['optimizer']

        # Convert demonstrations to tensor
        demo_tensor = torch.FloatTensor(np.array(demonstrations))

        for epoch in range(100):
            optimizer.zero_grad()

            # Encode and decode
            encoded = model['encoder'](demo_tensor)
            decoded = model['decoder'](encoded)

            # Reconstruction loss
            loss = nn.MSELoss()(decoded, demo_tensor)
            loss.backward()

            optimizer.step()

        # Return the encoded representation
        with torch.no_grad():
            encoded = model['encoder'](demo_tensor)

        return encoded

    def transfer_skill(self, source_skill: str, target_skill: str,
                      demonstrations: List[np.ndarray]) -> torch.Tensor:
        """Transfer skill from source to target"""
        # Encode source skill
        source_encoded = self.encode_skill(source_skill, demonstrations)

        # If target skill doesn't exist, create it
        if target_skill not in self.skill_encoders:
            self.encode_skill(target_skill, demonstrations)

        # Use the source encoder to encode target demonstrations
        target_tensor = torch.FloatTensor(np.array(demonstrations))

        with torch.no_grad():
            # Use source encoder to get latent representation
            if source_skill in self.transfer_models:
                source_encoder = self.transfer_models[source_skill]['encoder']
                target_encoded = source_encoder(target_tensor)

                # Decode using target decoder
                target_decoder = self.transfer_models[target_skill]['decoder']
                reconstructed = target_decoder(target_encoded)

        return reconstructed

# Example usage
if __name__ == "__main__":
    # Create multi-task learning system
    multitask_learner = MultiTaskRobotLearner(input_size=10, feature_size=128)

    # Add tasks
    multitask_learner.add_task('object_recognition', output_size=5)  # 5 object classes
    multitask_learner.add_task('pose_estimation', output_size=6)     # x,y,z,roll,pitch,yaw

    # Simulate training data
    X = torch.randn(100, 10)
    y_obj = torch.randint(0, 5, (100,))  # Object classes
    y_pose = torch.randn(100, 6)         # Pose data

    print("Training multi-task system...")
    multitask_learner.train_task('object_recognition', X, y_obj, epochs=5)
    multitask_learner.train_task('pose_estimation', X, y_pose, epochs=5)

    # Transfer to new task
    print("Transferring to new task...")
    multitask_learner.transfer_to_new_task('object_recognition', 'grasp_prediction', output_size=2)

    # Create skill transfer system
    skill_transfer = RobotSkillTransfer()

    # Simulate skill demonstrations
    demonstrations = [np.random.randn(50, 7) for _ in range(5)]  # 5 demonstrations of 7-DOF trajectory

    print("Encoding and transferring skills...")
    encoded_skill = skill_transfer.encode_skill('pick_place', demonstrations)
    print(f"Encoded skill shape: {encoded_skill.shape}")

    # Transfer skill
    transferred_skill = skill_transfer.transfer_skill('pick_place', 'place_only', demonstrations)
    print(f"Transferred skill shape: {transferred_skill.shape}")

    print("Transfer learning system test completed!")
```

## Adaptive Control and Online Learning

### Online Learning Algorithms

```python
#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Any
import threading
import time
from collections import deque

class OnlineLearner:
    def __init__(self, feature_size: int, lr: float = 0.01):
        """Initialize online learning system"""
        self.feature_size = feature_size
        self.lr = lr

        # Initialize weights
        self.weights = np.random.randn(feature_size) * 0.01
        self.bias = 0.0

        # For adaptive learning rate
        self.iteration = 0
        self.loss_history = deque(maxlen=100)

    def predict(self, features: np.ndarray) -> float:
        """Make prediction using current weights"""
        return np.dot(features, self.weights) + self.bias

    def update(self, features: np.ndarray, target: float) -> float:
        """Update weights using single example"""
        prediction = self.predict(features)
        error = target - prediction

        # Update weights using gradient descent
        self.weights += self.lr * error * features
        self.bias += self.lr * error

        # Store loss for adaptive learning rate
        loss = 0.5 * error ** 2
        self.loss_history.append(loss)

        self.iteration += 1

        # Adaptive learning rate based on recent performance
        if len(self.loss_history) > 10:
            recent_avg_loss = np.mean(list(self.loss_history)[-10:])
            if recent_avg_loss < 0.01:  # Converged
                self.lr = max(0.0001, self.lr * 0.99)  # Slow down learning
            elif recent_avg_loss > 0.1:  # Diverging
                self.lr = min(0.1, self.lr * 1.01)  # Speed up learning

        return loss

class AdaptiveNeuralNetwork:
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        """Initialize adaptive neural network"""
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        # Build network dynamically
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)
        self.optimizer = optim.SGD(self.network.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.network(x)

    def update(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Update network with single example"""
        self.optimizer.zero_grad()

        output = self.forward(x)
        loss = self.criterion(output, y)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def adapt_structure(self, performance_threshold: float = 0.1):
        """Adapt network structure based on performance"""
        # This is a simplified example - in practice, this would involve
        # adding/removing neurons or layers based on performance metrics
        pass

class ContinualLearningSystem:
    def __init__(self):
        """Initialize continual learning system"""
        self.online_learners = {}
        self.neural_networks = {}
        self.task_history = []
        self.current_task = None

        # Catastrophic forgetting prevention
        self.episodic_memory = deque(maxlen=1000)
        self.context_vectors = {}

    def register_task(self, task_name: str, input_size: int, output_size: int):
        """Register a new task"""
        self.online_learners[task_name] = OnlineLearner(input_size)
        self.neural_networks[task_name] = AdaptiveNeuralNetwork(
            input_size, [64, 32], output_size
        )
        self.context_vectors[task_name] = np.random.randn(input_size)

    def learn(self, task_name: str, features: np.ndarray, target: float):
        """Learn from a single example"""
        if task_name not in self.online_learners:
            raise ValueError(f"Task {task_name} not registered")

        # Store in episodic memory to prevent forgetting
        self.episodic_memory.append((task_name, features, target))

        # Update both online learner and neural network
        online_loss = self.online_learners[task_name].update(features, target)

        # Convert to tensors for neural network
        x_tensor = torch.FloatTensor(features).unsqueeze(0)
        y_tensor = torch.FloatTensor([target]).unsqueeze(0)

        nn_loss = self.neural_networks[task_name].update(x_tensor, y_tensor)

        # Store learning event
        learning_event = {
            'task': task_name,
            'features': features,
            'target': target,
            'online_loss': online_loss,
            'nn_loss': nn_loss,
            'timestamp': time.time()
        }

        self.task_history.append(learning_event)

        return online_loss, nn_loss

    def replay_memory(self, batch_size: int = 32):
        """Replay past experiences to prevent forgetting"""
        if len(self.episodic_memory) < batch_size:
            return

        # Sample random experiences from memory
        samples = list(self.episodic_memory)
        np.random.shuffle(samples)
        batch = samples[:batch_size]

        for task_name, features, target in batch:
            # Relearn from past experience
            self.learn(task_name, features, target)

    def switch_task(self, task_name: str):
        """Switch to a different task"""
        self.current_task = task_name

        # Replay some experiences from other tasks to maintain performance
        self.replay_memory(batch_size=10)

class RobotAdaptiveController:
    def __init__(self, state_size: int, action_size: int):
        """Initialize adaptive robot controller"""
        self.state_size = state_size
        self.action_size = action_size

        # Adaptive control components
        self.continual_learner = ContinualLearningSystem()
        self.policy_network = AdaptiveNeuralNetwork(
            state_size + action_size, [128, 64], 1  # Q-value prediction
        )

        # Initialize for different control tasks
        self.continual_learner.register_task('navigation', state_size, 2)  # x, y
        self.continual_learner.register_task('manipulation', state_size, action_size)
        self.continual_learner.register_task('balance', state_size, 1)  # stability

        # Control parameters
        self.control_history = deque(maxlen=1000)
        self.performance_metrics = {}

        # Start adaptive learning thread
        self.learning_active = True
        self.learning_thread = threading.Thread(target=self._adaptive_learning_loop)
        self.learning_thread.daemon = True
        self.learning_thread.start()

    def compute_control(self, state: np.ndarray, task: str = 'navigation') -> np.ndarray:
        """Compute control action based on current state and task"""
        # Use the appropriate learner for the task
        if task in self.continual_learner.online_learners:
            # Get prediction from online learner
            target = self.continual_learner.online_learners[task].predict(state)

            # Generate control action based on target
            if task == 'navigation':
                # Simple proportional controller for navigation
                control_action = target * 0.1  # Scale appropriately
            elif task == 'manipulation':
                # For manipulation, use neural network
                state_action = np.concatenate([state, np.zeros(self.action_size)])
                state_action_tensor = torch.FloatTensor(state_action).unsqueeze(0)
                q_value = self.policy_network.forward(state_action_tensor)
                control_action = np.random.randn(self.action_size) * 0.1  # Exploration
            else:
                control_action = np.zeros(self.action_size)
        else:
            # Default control
            control_action = np.zeros(self.action_size)

        # Store control history
        control_record = {
            'state': state.copy(),
            'action': control_action.copy(),
            'task': task,
            'timestamp': time.time()
        }
        self.control_history.append(control_record)

        return control_action

    def update_model(self, state: np.ndarray, action: np.ndarray,
                    reward: float, next_state: np.ndarray, task: str):
        """Update model based on experience"""
        # Learn from the experience
        features = np.concatenate([state, action])
        target = reward

        # Update the appropriate task
        try:
            self.continual_learner.learn(task, features, target)
        except ValueError:
            # Task not registered, register it
            self.continual_learner.register_task(task, len(features), 1)
            self.continual_learner.learn(task, features, target)

    def evaluate_performance(self, task: str) -> Dict:
        """Evaluate performance on a specific task"""
        if task not in self.performance_metrics:
            self.performance_metrics[task] = {
                'success_count': 0,
                'total_count': 0,
                'average_reward': 0.0,
                'recent_rewards': deque(maxlen=100)
            }

        metrics = self.performance_metrics[task]

        # Calculate performance metrics
        success_rate = metrics['success_count'] / max(1, metrics['total_count'])
        avg_reward = np.mean(metrics['recent_rewards']) if metrics['recent_rewards'] else 0.0

        return {
            'success_rate': success_rate,
            'average_reward': avg_reward,
            'total_episodes': metrics['total_count']
        }

    def _adaptive_learning_loop(self):
        """Background thread for continuous adaptation"""
        while self.learning_active:
            # Periodically replay experiences to prevent forgetting
            if len(self.continual_learner.episodic_memory) > 100:
                self.continual_learner.replay_memory(batch_size=32)

            # Update control parameters based on recent performance
            # This is where you would implement meta-learning algorithms
            # to adapt learning rates, network architectures, etc.

            time.sleep(1.0)  # Update every second

    def stop_learning(self):
        """Stop the adaptive learning process"""
        self.learning_active = False
        if self.learning_thread.is_alive():
            self.learning_thread.join()

# Example usage
if __name__ == "__main__":
    # Create adaptive controller
    controller = RobotAdaptiveController(state_size=10, action_size=6)

    print("Testing adaptive robot controller...")

    # Simulate robot interaction
    for episode in range(100):
        # Simulate state (e.g., sensor readings)
        state = np.random.randn(10)

        # Compute control for different tasks
        nav_control = controller.compute_control(state, 'navigation')
        manip_control = controller.compute_control(state, 'manipulation')

        # Simulate environment response and reward
        reward = np.random.randn()  # Random reward for simulation

        # Update model with experience
        controller.update_model(state, nav_control, reward, state, 'navigation')

        if episode % 20 == 0:
            # Evaluate performance
            nav_metrics = controller.evaluate_performance('navigation')
            print(f"Episode {episode}: Navigation success rate: {nav_metrics['success_rate']:.2f}")

    # Test different tasks
    controller.continual_learner.switch_task('manipulation')

    print("Adaptive learning system test completed!")

    # Clean up
    controller.stop_learning()
```

## Practical Exercises

### Exercise 1: Implement a Deep Reinforcement Learning System

**Objective**: Create a complete deep RL system for robot control that can learn complex behaviors.

**Steps**:
1. Implement DQN or PPO algorithm
2. Create a simulated robot environment
3. Train the agent on navigation or manipulation tasks
4. Evaluate learning performance and convergence
5. Test transfer to new environments

**Expected Outcome**: A functional deep RL system that can learn robot behaviors through interaction with the environment.

### Exercise 2: Design a Transfer Learning Framework

**Objective**: Develop a system that can transfer learned skills between different robotic tasks.

**Steps**:
1. Implement feature extraction networks
2. Create task-specific modules
3. Design transfer mechanisms
4. Test skill transfer between related tasks
5. Evaluate transfer effectiveness

**Expected Outcome**: A transfer learning system that can adapt previously learned skills to new but related tasks.

### Exercise 3: Adaptive Control System

**Objective**: Create an adaptive control system that learns and improves over time.

**Steps**:
1. Implement online learning algorithms
2. Create adaptive control mechanisms
3. Design performance evaluation metrics
4. Test adaptation to changing conditions
5. Evaluate long-term learning performance

**Expected Outcome**: An adaptive control system that improves its performance through continuous learning and adaptation.

## Chapter Summary

This chapter covered the essential concepts of robot learning and adaptation:

1. **Supervised Learning**: Implementing classification and regression models for robot perception and control tasks.

2. **Deep Learning**: Using neural networks for complex perception and decision-making in robotics.

3. **Reinforcement Learning**: Implementing Q-learning, policy gradients, and actor-critic methods for robot control.

4. **Transfer Learning**: Techniques for transferring knowledge between tasks and domains.

5. **Online Learning**: Adaptive systems that learn and improve during operation.

The integration of learning algorithms with robotic systems enables robots to adapt to new situations, improve their performance over time, and handle uncertainties in real-world environments. Modern robotics increasingly relies on these learning capabilities to achieve robust and flexible behavior.

## Further Reading

1. "Reinforcement Learning: An Introduction" by Sutton and Barto - Foundational RL text
2. "Deep Learning" by Goodfellow, Bengio, and Courville - Comprehensive deep learning reference
3. "Robot Learning" by Kober, Bagnell, and Peters - Robotics-specific learning approaches
4. "Transfer Learning for Robotics" by Taylor and Stone - Specialized transfer learning
5. "Continuous Control with Deep Reinforcement Learning" by Lillicrap et al. - DDPG and continuous control

## Assessment Questions

1. Compare different reinforcement learning algorithms for robotic applications and discuss their trade-offs.

2. Implement a deep Q-network for a robotic navigation task and analyze its performance.

3. Design a transfer learning system that can adapt manipulation skills to new objects.

4. Evaluate the effectiveness of different exploration strategies in robotic reinforcement learning.

5. Discuss the challenges of applying deep learning to real robotic systems.

6. Implement a policy gradient method for continuous control in robotics.

7. Analyze the impact of sample efficiency on robotic learning systems.

8. Design an online learning system for adaptive robot control.

9. Compare model-based vs model-free learning approaches for robotics.

10. Discuss the safety considerations in deploying learning robots in real-world environments.

