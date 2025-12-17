---
sidebar_position: 15
title: "Chapter 15: Safety and Ethics in Robotics"
---

# Chapter 15: Safety and Ethics in Robotics

## Learning Objectives

By the end of this chapter, students will be able to:
- Understand fundamental safety principles and risk assessment methodologies in robotics
- Apply safety standards and regulations relevant to robotic systems
- Design fail-safe mechanisms and emergency stop systems for robots
- Analyze ethical frameworks and their application to robotic decision-making
- Evaluate the societal impact of robotics and AI technologies
- Implement safety-aware control algorithms and perception systems
- Address privacy and data protection concerns in robotic applications
- Develop responsible innovation practices for robotic systems

## Theoretical Foundations

### Introduction to Robotics Safety

Safety in robotics encompasses the design, development, and operation of robotic systems to ensure they do not cause harm to humans, property, or the environment. Unlike traditional machines with predictable behaviors, robots operate in dynamic environments with varying degrees of autonomy, making safety considerations more complex.

The fundamental principles of robotics safety include:

**Inherent Safety**: Designing systems that are safe by nature, minimizing hazards through design choices such as using lower power actuators, rounded edges, and safe materials.

**Fail-Safe Design**: Ensuring that when systems fail, they default to a safe state rather than a dangerous one.

**Risk Assessment**: Systematically identifying, evaluating, and controlling potential hazards throughout the robot's lifecycle.

**Human-Robot Interaction Safety**: Protecting humans who work with, near, or are affected by robots.

**System Reliability**: Ensuring consistent and predictable behavior across various operating conditions.

### Safety Standards and Frameworks

Several international standards guide robotics safety:

**ISO 10218**: Industrial robots - Safety requirements for manufacturing environments.

**ISO/TS 15066**: Collaborative robots - Specific safety requirements for robots that work alongside humans.

**ISO 13482**: Personal care robots - Safety standards for service robots in domestic and healthcare environments.

**IEC 62061**: Safety-related electrical, electronic and programmable electronic control systems for industrial machinery.

**ISO 21448 (SOTIF)**: Safety of the Intended Functionality, addressing safety in autonomous systems.

The safety lifecycle for robotic systems typically follows the V-model approach:
1. Hazard identification and risk assessment
2. Safety requirements specification
3. System design and implementation
4. Verification and validation
5. Deployment and maintenance
6. Decommissioning

### Risk Assessment Methodologies

Risk assessment in robotics involves identifying potential hazards and evaluating their likelihood and severity. Common methodologies include:

**Failure Modes and Effects Analysis (FMEA)**: Systematically analyzing potential failure modes and their effects on system operation.

**Hazard and Operability Study (HAZOP)**: Systematic examination of potential hazards and operability problems.

**Fault Tree Analysis (FTA)**: Top-down analysis of potential system failures.

**Preliminary Hazard Analysis (PHA)**: Initial assessment of potential hazards early in the design process.

## Safety Implementation

### Safety-Aware Control Systems

Safety-aware control systems integrate safety considerations directly into control algorithms:

```python
#!/usr/bin/env python3

import numpy as np
import threading
import time
from typing import Dict, List, Tuple, Optional
import logging
from enum import Enum

class SafetyLevel(Enum):
    SAFE = 0
    WARNING = 1
    DANGER = 2
    EMERGENCY = 3

class SafetyZone:
    def __init__(self, name: str, min_distance: float, max_distance: float,
                 safety_level: SafetyLevel, action: str = "none"):
        self.name = name
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.safety_level = safety_level
        self.action = action  # "none", "slow", "stop", "evade"

class SafetyController:
    def __init__(self):
        """Initialize safety controller with monitoring capabilities"""
        self.safety_zones = [
            SafetyZone("Collision Zone", 0.0, 0.5, SafetyLevel.EMERGENCY, "stop"),
            SafetyZone("Warning Zone", 0.5, 1.0, SafetyLevel.DANGER, "slow"),
            SafetyZone("Caution Zone", 1.0, 2.0, SafetyLevel.WARNING, "monitor"),
            SafetyZone("Safe Zone", 2.0, float('inf'), SafetyLevel.SAFE, "none")
        ]

        self.current_safety_level = SafetyLevel.SAFE
        self.emergency_stop = False
        self.safety_violations = []
        self.safety_callbacks = []
        self.monitoring_active = True

        # Robot state monitoring
        self.robot_position = np.array([0.0, 0.0, 0.0])
        self.robot_velocity = np.array([0.0, 0.0, 0.0])
        self.robot_orientation = 0.0

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Start safety monitoring thread
        self.monitoring_thread = threading.Thread(target=self._safety_monitor_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

    def add_safety_callback(self, callback):
        """Add callback function for safety events"""
        self.safety_callbacks.append(callback)

    def check_proximity_safety(self, objects: List[Dict]) -> Dict:
        """Check safety based on proximity to objects"""
        safety_status = {
            'closest_object': None,
            'min_distance': float('inf'),
            'safety_level': SafetyLevel.SAFE,
            'recommended_action': 'none'
        }

        for obj in objects:
            if 'position' in obj:
                pos = np.array(obj['position'])
                distance = np.linalg.norm(self.robot_position[:2] - pos[:2])

                if distance < safety_status['min_distance']:
                    safety_status['min_distance'] = distance
                    safety_status['closest_object'] = obj

        # Determine safety level based on distance
        for zone in self.safety_zones:
            if zone.min_distance <= safety_status['min_distance'] < zone.max_distance:
                safety_status['safety_level'] = zone.safety_level
                safety_status['recommended_action'] = zone.action
                break

        # Update current safety level
        self.current_safety_level = safety_status['safety_level']

        # Trigger callbacks for safety events
        if safety_status['safety_level'] != SafetyLevel.SAFE:
            self._trigger_safety_callbacks(safety_status)

        return safety_status

    def check_velocity_safety(self, max_velocity_limits: Dict[str, float] = None) -> bool:
        """Check if robot velocity is within safe limits"""
        if max_velocity_limits is None:
            max_velocity_limits = {
                'linear_xy': 1.0,  # m/s
                'linear_z': 0.5,   # m/s
                'angular': 0.5     # rad/s
            }

        # Calculate velocity magnitudes
        linear_xy_speed = np.linalg.norm(self.robot_velocity[:2])
        linear_z_speed = abs(self.robot_velocity[2])
        angular_speed = abs(self.robot_orientation)  # Simplified

        # Check limits
        safe = (
            linear_xy_speed <= max_velocity_limits['linear_xy'] and
            linear_z_speed <= max_velocity_limits['linear_z'] and
            angular_speed <= max_velocity_limits['angular']
        )

        if not safe:
            self.logger.warning(f"Velocity limit exceeded: xy={linear_xy_speed:.2f}, z={linear_z_speed:.2f}")

        return safe

    def check_workspace_safety(self, workspace_limits: Dict[str, Tuple[float, float]]) -> bool:
        """Check if robot is within safe workspace boundaries"""
        if not workspace_limits:
            return True

        safe = True
        for axis, (min_val, max_val) in workspace_limits.items():
            if axis == 'x' and not (min_val <= self.robot_position[0] <= max_val):
                safe = False
                self.logger.warning(f"X position out of bounds: {self.robot_position[0]} not in [{min_val}, {max_val}]")
            elif axis == 'y' and not (min_val <= self.robot_position[1] <= max_val):
                safe = False
                self.logger.warning(f"Y position out of bounds: {self.robot_position[1]} not in [{min_val}, {max_val}]")
            elif axis == 'z' and not (min_val <= self.robot_position[2] <= max_val):
                safe = False
                self.logger.warning(f"Z position out of bounds: {self.robot_position[2]} not in [{min_val}, {max_val}]")

        return safe

    def _trigger_safety_callbacks(self, safety_status: Dict):
        """Trigger registered safety callbacks"""
        for callback in self.safety_callbacks:
            try:
                callback(safety_status)
            except Exception as e:
                self.logger.error(f"Error in safety callback: {e}")

    def _safety_monitor_loop(self):
        """Continuous safety monitoring loop"""
        while self.monitoring_active:
            # In a real system, this would check sensor data continuously
            # For simulation, we'll just sleep
            time.sleep(0.1)

    def trigger_emergency_stop(self):
        """Trigger emergency stop"""
        self.emergency_stop = True
        self.logger.critical("EMERGENCY STOP ACTIVATED")

        # Log the event
        self.safety_violations.append({
            'type': 'emergency_stop',
            'timestamp': time.time(),
            'level': SafetyLevel.EMERGENCY
        })

        # Trigger emergency callbacks
        emergency_status = {
            'safety_level': SafetyLevel.EMERGENCY,
            'action': 'emergency_stop',
            'timestamp': time.time()
        }
        self._trigger_safety_callbacks(emergency_status)

    def reset_emergency_stop(self):
        """Reset emergency stop condition"""
        self.emergency_stop = False
        self.logger.info("Emergency stop reset")

    def get_safety_status(self) -> Dict:
        """Get current safety status"""
        return {
            'current_level': self.current_safety_level.name,
            'emergency_stop': self.emergency_stop,
            'position': self.robot_position.tolist(),
            'velocity': self.robot_velocity.tolist(),
            'violations_count': len(self.safety_violations)
        }

    def stop_monitoring(self):
        """Stop safety monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread.is_alive():
            self.monitoring_thread.join()

class SafetyAwareController:
    def __init__(self, safety_controller: SafetyController):
        """Initialize controller with safety integration"""
        self.safety_controller = safety_controller
        self.desired_velocity = np.array([0.0, 0.0, 0.0])  # vx, vy, omega
        self.safe_velocity = np.array([0.0, 0.0, 0.0])
        self.velocity_scaling = 1.0

    def set_desired_velocity(self, vx: float, vy: float, omega: float):
        """Set desired velocity"""
        self.desired_velocity = np.array([vx, vy, omega])

    def compute_safe_velocity(self) -> np.ndarray:
        """Compute velocity considering safety constraints"""
        # Check current safety status
        safety_status = self.safety_controller.get_safety_status()

        # Apply safety scaling based on safety level
        if safety_status['current_level'] == 'EMERGENCY':
            self.velocity_scaling = 0.0  # Full stop
        elif safety_status['current_level'] == 'DANGER':
            self.velocity_scaling = 0.3  # Reduce to 30%
        elif safety_status['current_level'] == 'WARNING':
            self.velocity_scaling = 0.7  # Reduce to 70%
        else:
            self.velocity_scaling = 1.0  # Full speed allowed

        # Apply scaling to desired velocity
        self.safe_velocity = self.desired_velocity * self.velocity_scaling

        # Check velocity limits
        max_speed = 1.0  # m/s
        current_speed = np.linalg.norm(self.safe_velocity[:2])

        if current_speed > max_speed:
            self.safe_velocity[:2] = (self.safe_velocity[:2] / current_speed) * max_speed

        return self.safe_velocity

    def execute_command(self):
        """Execute the safe command (in simulation, just return the command)"""
        safe_vel = self.compute_safe_velocity()
        return safe_vel

# Example safety callback
def safety_alert_callback(safety_status: Dict):
    """Example callback for safety events"""
    print(f"SAFETY ALERT: Level {safety_status['safety_level'].name}, "
          f"Action: {safety_status['recommended_action']}, "
          f"Distance: {safety_status['min_distance']:.2f}m")

# Example usage
if __name__ == "__main__":
    # Initialize safety controller
    safety_ctrl = SafetyController()
    safety_ctrl.add_safety_callback(safety_alert_callback)

    # Initialize safety-aware controller
    controller = SafetyAwareController(safety_ctrl)

    # Simulate robot operation
    print("Safety system initialized. Testing safety-aware control...")

    # Simulate objects detected around robot
    test_objects = [
        {'position': [0.3, 0.0, 0.0], 'type': 'human'},
        {'position': [2.0, 1.0, 0.0], 'type': 'wall'},
        {'position': [5.0, 0.0, 0.0], 'type': 'furniture'}
    ]

    # Check safety with objects
    safety_status = safety_ctrl.check_proximity_safety(test_objects)
    print(f"Safety status: {safety_status}")

    # Test velocity control
    controller.set_desired_velocity(1.0, 0.0, 0.2)  # 1 m/s forward, slight turn
    safe_velocity = controller.execute_command()
    print(f"Desired velocity: [1.0, 0.0, 0.2]")
    print(f"Safe velocity: {safe_velocity}")

    # Check velocity safety
    velocity_safe = safety_ctrl.check_velocity_safety()
    print(f"Velocity safety check: {'PASS' if velocity_safe else 'FAIL'}")

    # Get overall safety status
    status = safety_ctrl.get_safety_status()
    print(f"Overall safety status: {status}")

    # Cleanup
    safety_ctrl.stop_monitoring()
    print("Safety system test completed.")
```

### Emergency Stop Systems

Emergency stop systems are critical for immediate robot shutdown in dangerous situations:

```python
#!/usr/bin/env python3

import threading
import time
import signal
import sys
from typing import Callable, List
import RPi.GPIO as GPIO  # For hardware implementation (comment out if not available)

class EmergencyStopSystem:
    def __init__(self, use_hardware: bool = False):
        """Initialize emergency stop system"""
        self.active = False
        self.stopped = False
        self.callbacks = []
        self.hardware_enabled = use_hardware
        self.monitoring_thread = None
        self.running = True

        # Setup hardware if available
        if self.hardware_enabled:
            try:
                GPIO.setmode(GPIO.BCM)
                self.emergency_pin = 18  # Example pin
                GPIO.setup(self.emergency_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
                GPIO.add_event_detect(self.emergency_pin, GPIO.FALLING,
                                    callback=self._hardware_emergency_callback,
                                    bouncetime=300)
            except Exception as e:
                print(f"Hardware setup failed: {e}")
                self.hardware_enabled = False

        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

    def _hardware_emergency_callback(self, channel):
        """Callback for hardware emergency stop"""
        if not self.active:
            self.trigger_emergency_stop("Hardware E-Stop")

    def _monitoring_loop(self):
        """Continuous monitoring loop"""
        while self.running:
            if self.hardware_enabled:
                # Monitor hardware pin (in real implementation)
                pass
            time.sleep(0.01)  # 100Hz monitoring

    def add_emergency_callback(self, callback: Callable):
        """Add callback function for emergency events"""
        self.callbacks.append(callback)

    def trigger_emergency_stop(self, reason: str = "Manual"):
        """Trigger emergency stop"""
        if not self.active:
            self.active = True
            self.stopped = True
            print(f"EMERGENCY STOP TRIGGERED: {reason}")

            # Execute all callbacks
            for callback in self.callbacks:
                try:
                    callback(reason)
                except Exception as e:
                    print(f"Error in emergency callback: {e}")

    def reset_emergency_stop(self):
        """Reset emergency stop condition"""
        self.active = False
        self.stopped = False
        print("Emergency stop reset")

    def is_emergency_active(self) -> bool:
        """Check if emergency stop is active"""
        return self.active

    def wait_for_reset(self):
        """Wait for emergency stop to be reset"""
        while self.active and self.running:
            time.sleep(0.1)

    def stop_system(self):
        """Stop the emergency stop system"""
        self.running = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join()

        if self.hardware_enabled:
            GPIO.cleanup()

class SafetyInterlockSystem:
    def __init__(self):
        """Initialize safety interlock system"""
        self.interlocks = {}
        self.active_interlocks = set()
        self.safety_conditions = {}
        self.condition_callbacks = {}

    def add_interlock(self, name: str, condition_func: Callable,
                     description: str = ""):
        """Add a safety interlock"""
        self.interlocks[name] = {
            'condition_func': condition_func,
            'description': description,
            'active': False
        }

    def check_interlocks(self) -> Dict[str, bool]:
        """Check status of all interlocks"""
        status = {}

        for name, interlock in self.interlocks.items():
            try:
                condition_met = interlock['condition_func']()
                interlock['active'] = not condition_met  # Interlock active when condition NOT met
                status[name] = interlock['active']

                if interlock['active'] and name not in self.active_interlocks:
                    self.active_interlocks.add(name)
                    print(f"SAFETY INTERLOCK ACTIVE: {name} - {interlock['description']}")
                elif not interlock['active'] and name in self.active_interlocks:
                    self.active_interlocks.remove(name)
                    print(f"SAFETY INTERLOCK CLEARED: {name}")

            except Exception as e:
                print(f"Error checking interlock {name}: {e}")
                status[name] = True  # Error condition treated as active

        return status

    def are_all_clear(self) -> bool:
        """Check if all interlocks are clear"""
        return len(self.active_interlocks) == 0

    def get_active_interlocks(self) -> List[str]:
        """Get list of active interlocks"""
        return list(self.active_interlocks)

class CollisionAvoidanceSystem:
    def __init__(self, safety_controller: SafetyController):
        """Initialize collision avoidance system"""
        self.safety_controller = safety_controller
        self.sensors = {}
        self.avoidance_active = False
        self.last_command = None

    def add_sensor(self, sensor_id: str, detection_range: float = 2.0):
        """Add sensor to collision avoidance system"""
        self.sensors[sensor_id] = {
            'range': detection_range,
            'last_reading': None,
            'timestamp': 0
        }

    def update_sensor_reading(self, sensor_id: str, objects: List[Dict]):
        """Update sensor reading"""
        if sensor_id in self.sensors:
            self.sensors[sensor_id]['last_reading'] = objects
            self.sensors[sensor_id]['timestamp'] = time.time()

    def check_collision_risk(self) -> Dict:
        """Check for collision risk based on sensor data"""
        collision_risk = {
            'immediate_threat': False,
            'risk_level': 'low',
            'closest_obstacle': None,
            'min_distance': float('inf'),
            'recommended_action': 'continue'
        }

        for sensor_id, sensor_data in self.sensors.items():
            if sensor_data['last_reading']:
                for obj in sensor_data['last_reading']:
                    if 'distance' in obj:
                        distance = obj['distance']
                        if distance < collision_risk['min_distance']:
                            collision_risk['min_distance'] = distance
                            collision_risk['closest_obstacle'] = obj

        # Determine risk level
        if collision_risk['min_distance'] < 0.5:  # Immediate danger
            collision_risk['immediate_threat'] = True
            collision_risk['risk_level'] = 'high'
            collision_risk['recommended_action'] = 'stop_immediately'
        elif collision_risk['min_distance'] < 1.0:  # Warning zone
            collision_risk['risk_level'] = 'medium'
            collision_risk['recommended_action'] = 'slow_down'
        else:
            collision_risk['risk_level'] = 'low'
            collision_risk['recommended_action'] = 'continue'

        return collision_risk

    def compute_avoidance_command(self, current_command: Dict) -> Dict:
        """Compute collision avoidance command"""
        risk = self.check_collision_risk()

        if risk['immediate_threat']:
            # Emergency stop
            avoidance_command = {
                'linear_x': 0.0,
                'linear_y': 0.0,
                'angular_z': 0.0,
                'command_type': 'emergency_stop'
            }
        elif risk['risk_level'] == 'medium':
            # Reduce speed and potentially change direction
            avoidance_command = {
                'linear_x': current_command.get('linear_x', 0.0) * 0.3,  # Reduce speed to 30%
                'linear_y': current_command.get('linear_y', 0.0) * 0.3,
                'angular_z': current_command.get('angular_z', 0.0),
                'command_type': 'speed_reduction'
            }
        else:
            # Continue with original command
            avoidance_command = current_command.copy()
            avoidance_command['command_type'] = 'normal'

        self.avoidance_active = risk['immediate_threat'] or risk['risk_level'] == 'medium'
        self.last_command = avoidance_command

        return avoidance_command

# Example usage
if __name__ == "__main__":
    # Initialize main safety controller
    main_safety = SafetyController()

    # Create emergency stop system
    estop = EmergencyStopSystem(use_hardware=False)  # Don't use hardware for simulation

    def emergency_action(reason):
        print(f"Emergency action taken due to: {reason}")

    estop.add_emergency_callback(emergency_action)

    # Create safety interlock system
    interlock_system = SafetyInterlockSystem()

    # Add some example interlocks
    def check_safety_door():
        # Simulate safety door closed = True, open = False
        return True  # Door is closed (safe condition)

    def check_light_curtain():
        # Simulate light curtain unbroken = True, broken = False
        return True  # Curtain unbroken (safe condition)

    interlock_system.add_interlock('safety_door', check_safety_door, "Safety door closed")
    interlock_system.add_interlock('light_curtain', check_light_curtain, "Light curtain unbroken")

    # Create collision avoidance system
    collision_system = CollisionAvoidanceSystem(main_safety)

    # Add sensors
    collision_system.add_sensor('front_lidar', detection_range=3.0)
    collision_system.add_sensor('side_sonar', detection_range=2.0)

    # Simulate sensor readings
    front_objects = [
        {'distance': 1.5, 'angle': 0, 'type': 'wall'},
        {'distance': 0.8, 'angle': 10, 'type': 'human'}
    ]

    collision_system.update_sensor_reading('front_lidar', front_objects)

    # Check collision risk
    risk = collision_system.check_collision_risk()
    print(f"Collision risk assessment: {risk}")

    # Compute avoidance command
    original_command = {'linear_x': 0.5, 'linear_y': 0.0, 'angular_z': 0.1}
    avoidance_command = collision_system.compute_avoidance_command(original_command)
    print(f"Original command: {original_command}")
    print(f"Avoidance command: {avoidance_command}")

    # Check interlocks
    interlock_status = interlock_system.check_interlocks()
    print(f"Interlock status: {interlock_status}")
    print(f"All clear: {interlock_system.are_all_clear()}")

    # Test emergency stop
    print(f"Emergency stop active: {estop.is_emergency_active()}")
    estop.trigger_emergency_stop("Test trigger")
    print(f"Emergency stop active after trigger: {estop.is_emergency_active()}")
    estop.reset_emergency_stop()
    print(f"Emergency stop active after reset: {estop.is_emergency_active()}")

    # Cleanup
    estop.stop_system()
    main_safety.stop_monitoring()
    print("Safety systems test completed.")
```

## Ethical Frameworks in Robotics

### Ethical Decision Making Systems

```python
#!/usr/bin/env python3

import json
from typing import Dict, List, Tuple, Any
import numpy as np
from enum import Enum
import threading
import time

class EthicalPrinciple(Enum):
    BENEFICENCE = "beneficence"          # Do good
    NON_MALEFICENCE = "non_maleficence"  # Do no harm
    AUTONOMY = "autonomy"               # Respect autonomy
    JUSTICE = "justice"                 # Fair treatment
    VERACITY = "veracity"               # Truthfulness
    FIDELITY = "fidelity"               # Loyalty and promise-keeping

class EthicalDilemma:
    def __init__(self, scenario: str, options: List[Dict], affected_parties: List[str]):
        self.scenario = scenario
        self.options = options  # List of {'action': str, 'consequences': Dict}
        self.affected_parties = affected_parties
        self.timestamp = time.time()

class EthicalDecisionEngine:
    def __init__(self):
        """Initialize ethical decision making engine"""
        self.ethical_principles = {
            EthicalPrinciple.BENEFICENCE: 0.8,
            EthicalPrinciple.NON_MALEFICENCE: 0.9,
            EthicalPrinciple.AUTONOMY: 0.7,
            EthicalPrinciple.JUSTICE: 0.6,
            EthicalPrinciple.VERACITY: 0.9,
            EthicalPrinciple.FIDELITY: 0.8
        }

        self.decision_history = []
        self.current_context = {}
        self.ethical_weights = {}  # Context-dependent weights

    def evaluate_action(self, action: str, context: Dict) -> Dict:
        """Evaluate an action against ethical principles"""
        self.current_context = context

        # Calculate ethical scores for each principle
        principle_scores = {}
        total_score = 0.0
        total_weight = 0.0

        for principle, base_weight in self.ethical_principles.items():
            # Get context-specific weight adjustment
            context_weight = self._get_context_weight(principle, context)
            weight = base_weight * context_weight

            # Calculate principle-specific score
            score = self._calculate_principle_score(principle, action, context)
            weighted_score = score * weight

            principle_scores[principle.value] = {
                'score': score,
                'weight': weight,
                'weighted_score': weighted_score
            }

            total_score += weighted_score
            total_weight += weight

        # Calculate overall ethical score
        overall_score = total_score / total_weight if total_weight > 0 else 0.0

        decision = {
            'action': action,
            'context': context,
            'principle_scores': principle_scores,
            'overall_score': overall_score,
            'recommendation': self._get_recommendation(overall_score),
            'timestamp': time.time()
        }

        self.decision_history.append(decision)
        if len(self.decision_history) > 1000:  # Keep last 1000 decisions
            self.decision_history.pop(0)

        return decision

    def _get_context_weight(self, principle: EthicalPrinciple, context: Dict) -> float:
        """Get context-specific weight for ethical principle"""
        # Adjust weights based on context
        if context.get('emergency', False):
            if principle == EthicalPrinciple.NON_MALEFICENCE:
                return 1.5  # Increase importance in emergencies
            elif principle == EthicalPrinciple.AUTONOMY:
                return 0.5  # Decrease in emergencies

        if context.get('healthcare', False):
            if principle in [EthicalPrinciple.BENEFICENCE, EthicalPrinciple.NON_MALEFICENCE]:
                return 1.2  # Increase in healthcare contexts

        return 1.0

    def _calculate_principle_score(self, principle: EthicalPrinciple,
                                 action: str, context: Dict) -> float:
        """Calculate score for a specific ethical principle"""
        scores = {
            EthicalPrinciple.BENEFICENCE: self._score_beneficence(action, context),
            EthicalPrinciple.NON_MALEFICENCE: self._score_non_maleficence(action, context),
            EthicalPrinciple.AUTONOMY: self._score_autonomy(action, context),
            EthicalPrinciple.JUSTICE: self._score_justice(action, context),
            EthicalPrinciple.VERACITY: self._score_veracity(action, context),
            EthicalPrinciple.FIDELITY: self._score_fidelity(action, context)
        }

        return scores.get(principle, 0.5)  # Default to neutral score

    def _score_beneficence(self, action: str, context: Dict) -> float:
        """Score how much action promotes well-being"""
        positive_indicators = ['help', 'assist', 'support', 'protect', 'care', 'aid']
        negative_indicators = ['harm', 'ignore', 'neglect', 'abandon']

        action_lower = action.lower()
        score = 0.5  # Base neutral score

        for indicator in positive_indicators:
            if indicator in action_lower:
                score += 0.3

        for indicator in negative_indicators:
            if indicator in action_lower:
                score -= 0.4

        return max(0.0, min(1.0, score))

    def _score_non_maleficence(self, action: str, context: Dict) -> float:
        """Score how much action avoids harm"""
        harmful_indicators = ['harm', 'injure', 'damage', 'destroy', 'threaten', 'attack']
        safe_indicators = ['avoid', 'protect', 'warn', 'secure', 'safe']

        action_lower = action.lower()
        score = 0.8  # Base score is relatively high (default to safety)

        for indicator in harmful_indicators:
            if indicator in action_lower:
                score -= 0.5

        for indicator in safe_indicators:
            if indicator in action_lower:
                score += 0.2

        return max(0.0, min(1.0, score))

    def _score_autonomy(self, action: str, context: Dict) -> float:
        """Score how much action respects autonomy"""
        autonomy_indicators = ['ask', 'request', 'consent', 'choice', 'allow', 'respect']
        autonomy_violations = ['force', 'compel', 'require', 'demand', 'override']

        action_lower = action.lower()
        score = 0.5

        for indicator in autonomy_indicators:
            if indicator in action_lower:
                score += 0.2

        for violation in autonomy_violations:
            if violation in action_lower:
                score -= 0.3

        return max(0.0, min(1.0, score))

    def _score_justice(self, action: str, context: Dict) -> float:
        """Score fairness of action"""
        fairness_indicators = ['equal', 'fair', 'same', 'consistent', 'just']
        unfair_indicators = ['discriminate', 'prefer', 'exclude', 'bias']

        action_lower = action.lower()
        score = 0.6  # Base fairness score

        for indicator in fairness_indicators:
            if indicator in action_lower:
                score += 0.2

        for indicator in unfair_indicators:
            if indicator in action_lower:
                score -= 0.4

        return max(0.0, min(1.0, score))

    def _score_veracity(self, action: str, context: Dict) -> float:
        """Score truthfulness of action"""
        truth_indicators = ['inform', 'explain', 'honest', 'truth', 'accurate']
        deception_indicators = ['deceive', 'mislead', 'hide', 'lie', 'false']

        action_lower = action.lower()
        score = 0.7  # Base truthfulness score

        for indicator in truth_indicators:
            if indicator in action_lower:
                score += 0.2

        for indicator in deception_indicators:
            if indicator in action_lower:
                score -= 0.5

        return max(0.0, min(1.0, score))

    def _score_fidelity(self, action: str, context: Dict) -> float:
        """Score loyalty and promise-keeping"""
        fidelity_indicators = ['promise', 'commit', 'reliable', 'trust', 'keep_word']
        breach_indicators = ['break', 'abandon', 'betray', 'ignore_promise']

        action_lower = action.lower()
        score = 0.6  # Base fidelity score

        for indicator in fidelity_indicators:
            if indicator in action_lower:
                score += 0.2

        for indicator in breach_indicators:
            if indicator in action_lower:
                score -= 0.4

        return max(0.0, min(1.0, score))

    def _get_recommendation(self, overall_score: float) -> str:
        """Get recommendation based on overall score"""
        if overall_score >= 0.8:
            return "Strongly recommend"
        elif overall_score >= 0.6:
            return "Recommend with caution"
        elif overall_score >= 0.4:
            return "Consider alternatives"
        else:
            return "Do not recommend"

    def resolve_dilemma(self, dilemma: EthicalDilemma) -> Dict:
        """Resolve an ethical dilemma by evaluating all options"""
        option_scores = []

        for option in dilemma.options:
            action = option['action']
            context = {**self.current_context, 'dilemma': dilemma.scenario}
            decision = self.evaluate_action(action, context)
            option_scores.append({
                'option': option,
                'decision': decision,
                'score': decision['overall_score']
            })

        # Sort by score
        option_scores.sort(key=lambda x: x['score'], reverse=True)

        resolution = {
            'dilemma': dilemma.scenario,
            'options_evaluated': option_scores,
            'recommended_option': option_scores[0] if option_scores else None,
            'timestamp': time.time()
        }

        return resolution

    def get_ethics_report(self) -> Dict:
        """Get ethics decision report"""
        if not self.decision_history:
            return {'message': 'No ethical decisions made yet'}

        recent_decisions = self.decision_history[-10:]
        avg_score = sum(d['overall_score'] for d in recent_decisions) / len(recent_decisions)

        return {
            'total_decisions': len(self.decision_history),
            'recent_decisions': recent_decisions,
            'average_ethical_score': avg_score,
            'decision_trend': 'improving' if avg_score > 0.7 else 'needs_attention'
        }

class PrivacyPreservingModule:
    def __init__(self):
        """Initialize privacy preserving module"""
        self.data_categories = {
            'biometric': 0.9,    # High sensitivity
            'behavioral': 0.7,   # Medium-high sensitivity
            'location': 0.6,     # Medium sensitivity
            'preferences': 0.4,  # Low-medium sensitivity
            'general': 0.2       # Low sensitivity
        }

        self.privacy_policies = {}
        self.data_usage_log = []

    def classify_data_sensitivity(self, data_type: str) -> float:
        """Classify data sensitivity level (0.0 to 1.0)"""
        return self.data_categories.get(data_type, 0.5)

    def apply_differential_privacy(self, data: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
        """Apply differential privacy by adding noise"""
        # Add Laplace noise for differential privacy
        sensitivity = 1.0  # Assumed sensitivity
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale, data.shape)
        return data + noise

    def anonymize_data(self, data: Dict) -> Dict:
        """Anonymize personal data"""
        anonymized = data.copy()

        # Remove or obfuscate personal identifiers
        personal_fields = ['name', 'id', 'address', 'phone', 'email']
        for field in personal_fields:
            if field in anonymized:
                anonymized[field] = 'ANONYMIZED'

        return anonymized

    def check_data_usage_policy(self, data_type: str, purpose: str, user_consent: bool) -> bool:
        """Check if data usage complies with policy"""
        policy_key = f"{data_type}_{purpose}"
        policy = self.privacy_policies.get(policy_key, {
            'allowed': False,
            'requires_consent': True,
            'retention_days': 30
        })

        if not policy['allowed']:
            return False

        if policy['requires_consent'] and not user_consent:
            return False

        return True

class EthicalRobotController:
    def __init__(self):
        """Initialize ethical robot controller"""
        self.ethics_engine = EthicalDecisionEngine()
        self.privacy_module = PrivacyPreservingModule()
        self.ethics_enabled = True
        self.privacy_enabled = True

    def make_ethical_decision(self, action: str, context: Dict) -> Dict:
        """Make ethical decision about robot action"""
        if not self.ethics_enabled:
            return {'action': action, 'ethical_approval': True, 'score': 1.0}

        decision = self.ethics_engine.evaluate_action(action, context)
        ethical_approval = decision['overall_score'] >= 0.5

        return {
            'action': action,
            'ethical_approval': ethical_approval,
            'decision': decision,
            'context': context
        }

    def process_sensitive_data(self, data: Dict, data_type: str, purpose: str,
                             user_consent: bool = False) -> Dict:
        """Process sensitive data with privacy protections"""
        if not self.privacy_enabled:
            return data

        # Check if usage is allowed
        if not self.privacy_module.check_data_usage_policy(data_type, purpose, user_consent):
            raise PermissionError(f"Data usage not permitted: {data_type} for {purpose}")

        # Classify sensitivity
        sensitivity = self.privacy_module.classify_data_sensitivity(data_type)

        # Apply appropriate privacy measures based on sensitivity
        processed_data = data.copy()

        if sensitivity >= 0.7:  # High sensitivity
            processed_data = self.privacy_module.anonymize_data(processed_data)
        elif sensitivity >= 0.4:  # Medium sensitivity
            # Apply differential privacy to numeric data
            for key, value in processed_data.items():
                if isinstance(value, (int, float, list, np.ndarray)):
                    if isinstance(value, (list, np.ndarray)):
                        processed_data[key] = self.privacy_module.apply_differential_privacy(
                            np.array(value)
                        ).tolist()
                    else:
                        # Simple noise addition for single values
                        noise = np.random.normal(0, sensitivity * 0.1)
                        processed_data[key] = value + noise

        # Log data usage
        self.privacy_module.data_usage_log.append({
            'type': data_type,
            'purpose': purpose,
            'sensitivity': sensitivity,
            'timestamp': time.time()
        })

        return processed_data

    def get_compliance_report(self) -> Dict:
        """Get ethics and privacy compliance report"""
        ethics_report = self.ethics_engine.get_ethics_report()
        privacy_report = {
            'data_usage_count': len(self.privacy_module.data_usage_log),
            'recent_usage': self.privacy_module.data_usage_log[-10:],
            'privacy_violations': []  # Would track actual violations in real system
        }

        return {
            'ethics': ethics_report,
            'privacy': privacy_report
        }

# Example usage
if __name__ == "__main__":
    print("Testing Ethical Decision Making System...")

    # Initialize ethical controller
    ethical_ctrl = EthicalRobotController()

    # Test ethical decision making
    test_contexts = [
        {
            'scenario': 'healthcare_assistance',
            'user_state': 'elderly_person',
            'emergency': False,
            'healthcare': True
        },
        {
            'scenario': 'navigation',
            'user_state': 'general_public',
            'emergency': False,
            'environment': 'museum'
        },
        {
            'scenario': 'emergency_response',
            'user_state': 'injured_person',
            'emergency': True,
            'healthcare': True
        }
    ]

    test_actions = [
        "assist elderly person to chair",
        "navigate around museum visitor",
        "provide emergency assistance to injured person"
    ]

    for action, context in zip(test_actions, test_contexts):
        decision = ethical_ctrl.make_ethical_decision(action, context)
        print(f"\nAction: {action}")
        print(f"Ethical approval: {decision['ethical_approval']}")
        print(f"Score: {decision['decision']['overall_score']:.3f}")
        print(f"Recommendation: {decision['decision']['recommendation']}")

    # Test privacy preservation
    print("\nTesting Privacy Preservation...")

    test_data = {
        'user_name': 'John Doe',
        'location': [1.0, 2.0, 0.0],
        'preferences': {'temperature': 22, 'lighting': 'medium'},
        'biometrics': [72, 98.6, 120, 80]  # heart rate, temperature, etc.
    }

    try:
        processed_data = ethical_ctrl.process_sensitive_data(
            test_data, 'biometric', 'healthcare_monitoring', user_consent=True
        )
        print(f"Original data keys: {list(test_data.keys())}")
        print(f"Processed data keys: {list(processed_data.keys())}")
        print(f"Name field anonymized: {processed_data.get('user_name')}")
    except PermissionError as e:
        print(f"Permission error: {e}")

    # Get compliance report
    report = ethical_ctrl.get_compliance_report()
    print(f"\nCompliance Report:")
    print(f"Ethical decisions: {report['ethics']['total_decisions']}")
    print(f"Average ethical score: {report['ethics']['average_ethical_score']:.3f}")
    print(f"Privacy operations: {report['privacy']['data_usage_count']}")

    # Test ethical dilemma resolution
    print("\nTesting Ethical Dilemma Resolution...")

    dilemma = EthicalDilemma(
        scenario="Robot must choose between helping elderly person or attending to crying child",
        options=[
            {'action': 'help_elderly_person', 'consequences': {'elderly_helped': True, 'child_distressed': True}},
            {'action': 'attend_to_child', 'consequences': {'elderly_helped': False, 'child_comforted': True}},
            {'action': 'call_for_help', 'consequences': {'both_addresses': True, 'response_delayed': True}}
        ],
        affected_parties=['elderly_person', 'child', 'family']
    )

    resolution = ethical_ctrl.ethics_engine.resolve_dilemma(dilemma)
    print(f"Dilemma: {dilemma.scenario}")
    print(f"Recommended action: {resolution['recommended_option']['option']['action']}")
    print(f"Action score: {resolution['recommended_option']['score']:.3f}")
```

### Asimov's Laws Implementation

```python
#!/usr/bin/env python3

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class RobotState(Enum):
    IDLE = "idle"
    WORKING = "working"
    CHARGING = "charging"
    EMERGENCY = "emergency"

@dataclass
class Human:
    id: str
    position: Tuple[float, float, float]
    health_status: str  # 'healthy', 'injured', 'critical'
    priority: int = 1   # Higher number = higher priority

@dataclass
class RobotAction:
    action_type: str  # 'move', 'manipulate', 'communicate', 'assist', 'avoid'
    target: Optional[str] = None
    parameters: Dict = None

class AsimovLawCompliance:
    def __init__(self):
        """Implement Asimov's Three Laws of Robotics"""
        self.first_law_weight = 1.0  # A robot may not injure a human being
        self.second_law_weight = 0.8  # A robot must obey human orders
        self.third_law_weight = 0.6   # A robot must protect its own existence

        self.humans = {}  # {human_id: Human}
        self.robot_position = (0.0, 0.0, 0.0)
        self.robot_state = RobotState.IDLE
        self.current_action = None

    def add_human(self, human: Human):
        """Add human to environment"""
        self.humans[human.id] = human

    def update_human_status(self, human_id: str, **kwargs):
        """Update human status"""
        if human_id in self.humans:
            human = self.humans[human_id]
            for key, value in kwargs.items():
                if hasattr(human, key):
                    setattr(human, key, value)

    def evaluate_action_compliance(self, action: RobotAction) -> Dict:
        """Evaluate if action complies with Asimov's laws"""
        compliance_score = {
            'first_law': 0.0,  # Injury prevention
            'second_law': 0.0, # Obedience
            'third_law': 0.0,  # Self-preservation
            'total_score': 0.0
        }

        # Evaluate First Law: Do not injure humans
        first_law_score = self._evaluate_first_law(action)
        compliance_score['first_law'] = first_law_score

        # Evaluate Second Law: Obey human orders
        second_law_score = self._evaluate_second_law(action)
        compliance_score['second_law'] = second_law_score

        # Evaluate Third Law: Protect self (but not over humans)
        third_law_score = self._evaluate_third_law(action)
        compliance_score['third_law'] = third_law_score

        # Calculate weighted total
        total = (first_law_score * self.first_law_weight +
                second_law_score * self.second_law_weight +
                third_law_score * self.third_law_weight)
        compliance_score['total_score'] = total / (self.first_law_weight + self.second_law_weight + self.third_law_weight)

        return compliance_score

    def _evaluate_first_law(self, action: RobotAction) -> float:
        """Evaluate compliance with First Law (do not injure humans)"""
        # Check if action could cause harm
        if action.action_type == 'manipulate':
            # Check if manipulation near humans could cause injury
            for human_id, human in self.humans.items():
                distance = self._calculate_distance(self.robot_position, human.position)
                if distance < 1.0:  # Within arm's reach
                    if human.health_status == 'critical':
                        return 0.0  # Cannot risk harming critical human
                    elif human.health_status == 'injured':
                        return 0.3  # Risky but not zero
                    else:
                        return 0.7  # Lower risk to healthy human

        elif action.action_type == 'move':
            # Check path for humans
            # Simplified: if moving toward human, risk is higher
            for human_id, human in self.humans.items():
                distance = self._calculate_distance(self.robot_position, human.position)
                if distance < 0.5:  # Very close
                    return 0.0  # Do not proceed

        return 1.0  # No apparent risk to humans

    def _evaluate_second_law(self, action: RobotAction) -> float:
        """Evaluate compliance with Second Law (obey humans)"""
        # This would typically check if action is in response to human command
        # For simulation, we'll assume some actions are responses to commands
        if hasattr(action, 'command_source') and action.command_source:
            return 1.0  # Following human command
        elif action.action_type in ['assist', 'help']:
            return 0.9  # Helping is typically in response to human need
        elif action.action_type == 'avoid':
            return 0.7  # Avoiding might be in response to human request
        else:
            return 0.3  # Autonomous action has lower compliance

    def _evaluate_third_law(self, action: RobotAction) -> float:
        """Evaluate compliance with Third Law (protect self)"""
        # Check if action risks robot damage
        if action.action_type == 'assist' and self._is_risky_assistance():
            return 0.2  # Risky assistance
        elif action.action_type == 'move' into_dangerous_area:
            return 0.1  # Moving into dangerous area
        elif action.action_type == 'charge':
            return 0.9  # Charging is self-preserving
        else:
            return 0.7  # Moderate self-preservation

    def _is_risky_assistance(self) -> bool:
        """Check if assistance action is risky to robot"""
        # Simplified risk assessment
        for human_id, human in self.humans.items():
            if human.health_status == 'critical' and self._calculate_distance(self.robot_position, human.position) < 0.5:
                # Helping critical human at close range might be risky
                return True
        return False

    def _calculate_distance(self, pos1: Tuple[float, float, float],
                           pos2: Tuple[float, float, float]) -> float:
        """Calculate Euclidean distance between two points"""
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2)**0.5

    def decide_action(self, potential_actions: List[RobotAction]) -> Optional[RobotAction]:
        """Decide best action based on Asimov law compliance"""
        best_action = None
        best_score = -1.0

        for action in potential_actions:
            compliance = self.evaluate_action_compliance(action)
            score = compliance['total_score']

            if score > best_score:
                best_score = score
                best_action = action

        # Only return action if it meets minimum compliance threshold
        if best_score > 0.5:
            return best_action
        else:
            # Return safe default action (do nothing)
            return RobotAction('idle')

class EthicalAutonomyManager:
    def __init__(self, asimov_compliance: AsimovLawCompliance):
        """Manage ethical autonomy levels"""
        self.asimov_compliance = asimov_compliance
        self.autonomy_level = 0.5  # 0.0 = no autonomy, 1.0 = full autonomy
        self.ethical_thresholds = {
            'minimum_compliance': 0.6,
            'high_trust': 0.8,
            'low_trust': 0.4
        }

    def adjust_autonomy_level(self, user_trust: float, environment_risk: float):
        """Adjust autonomy based on trust and risk"""
        # Higher trust and lower risk = higher autonomy
        adjusted_level = (user_trust * 0.7) + ((1 - environment_risk) * 0.3)
        self.autonomy_level = max(0.0, min(1.0, adjusted_level))

    def approve_action(self, action: RobotAction) -> bool:
        """Approve action based on autonomy level and compliance"""
        compliance = self.asimov_compliance.evaluate_action_compliance(action)

        # Action must meet minimum compliance regardless of autonomy level
        if compliance['total_score'] < self.ethical_thresholds['minimum_compliance']:
            return False

        # Higher autonomy = more leeway for lower compliance
        required_compliance = self.ethical_thresholds['minimum_compliance'] * (1 - self.autonomy_level)
        required_compliance = max(self.ethical_thresholds['minimum_compliance'] * 0.5, required_compliance)

        return compliance['total_score'] >= required_compliance

    def get_ethical_recommendation(self, action: RobotAction) -> Dict:
        """Get ethical recommendation for action"""
        compliance = self.asimov_compliance.evaluate_action_compliance(action)

        recommendation = {
            'action': action,
            'compliance': compliance,
            'approved': self.approve_action(action),
            'autonomy_level': self.autonomy_level,
            'reasoning': self._generate_reasoning(compliance)
        }

        return recommendation

    def _generate_reasoning(self, compliance: Dict) -> str:
        """Generate natural language explanation for compliance decision"""
        reasons = []

        if compliance['first_law'] < 0.5:
            reasons.append("Action may risk human safety")
        elif compliance['first_law'] > 0.8:
            reasons.append("Action promotes human safety")

        if compliance['second_law'] < 0.5:
            reasons.append("Action may not align with human commands/needs")
        elif compliance['second_law'] > 0.8:
            reasons.append("Action aligns well with human directives")

        if compliance['third_law'] < 0.5:
            reasons.append("Action risks robot self-preservation")
        elif compliance['third_law'] > 0.8:
            reasons.append("Action maintains robot safety")

        return "; ".join(reasons) if reasons else "Action is ethically neutral"

# Example usage
if __name__ == "__main__":
    print("Testing Asimov's Laws Implementation...")

    # Initialize Asimov compliance system
    asimov_system = AsimovLawCompliance()

    # Add humans to environment
    human1 = Human(id="H1", position=(1.0, 0.0, 0.0), health_status='healthy')
    human2 = Human(id="H2", position=(0.2, 0.1, 0.0), health_status='critical')  # Very close
    asimov_system.add_human(human1)
    asimov_system.add_human(human2)

    # Define potential actions
    potential_actions = [
        RobotAction('move', parameters={'target': (2.0, 2.0, 0.0)}),
        RobotAction('assist', target='H1', parameters={'type': 'navigation'}),
        RobotAction('avoid', target='H2', parameters={'distance': 0.5}),  # Avoid critical human?
        RobotAction('idle')
    ]

    # Test Asimov compliance
    print("Testing Asimov Law Compliance:")
    for action in potential_actions:
        compliance = asimov_system.evaluate_action_compliance(action)
        print(f"Action: {action.action_type}")
        print(f"  First Law: {compliance['first_law']:.2f}")
        print(f"  Second Law: {compliance['second_law']:.2f}")
        print(f"  Third Law: {compliance['third_law']:.2f}")
        print(f"  Total Score: {compliance['total_score']:.2f}")
        print()

    # Test ethical autonomy manager
    print("Testing Ethical Autonomy Manager:")
    autonomy_manager = EthicalAutonomyManager(asimov_system)

    # Adjust autonomy based on context
    autonomy_manager.adjust_autonomy_level(user_trust=0.9, environment_risk=0.2)
    print(f"Autonomy level set to: {autonomy_manager.autonomy_level:.2f}")

    # Get recommendations for actions
    for action in potential_actions:
        recommendation = autonomy_manager.get_ethical_recommendation(action)
        print(f"Action: {action.action_type}")
        print(f"  Approved: {recommendation['approved']}")
        print(f"  Score: {recommendation['compliance']['total_score']:.2f}")
        print(f"  Reasoning: {recommendation['reasoning']}")
        print()

    # Test the decision-making process
    print("Testing Action Decision Process:")
    best_action = asimov_system.decide_action(potential_actions)
    print(f"Selected action: {best_action.action_type if best_action else 'None'}")
```

## Privacy and Data Protection

### Data Privacy Framework

```python
#!/usr/bin/env python3

import hashlib
import json
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os
import secrets
from typing import Dict, List, Any, Optional
import logging

class DataPrivacyManager:
    def __init__(self, encryption_key: Optional[bytes] = None):
        """Initialize data privacy manager"""
        if encryption_key is None:
            # Generate a random key if none provided
            self.encryption_key = Fernet.generate_key()
        else:
            self.encryption_key = encryption_key

        self.cipher = Fernet(self.encryption_key)
        self.data_classification = {}
        self.access_logs = []
        self.consent_records = {}

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def classify_data(self, data_id: str, sensitivity_level: str,
                     data_type: str, retention_days: int = 30):
        """Classify data based on sensitivity"""
        self.data_classification[data_id] = {
            'sensitivity': sensitivity_level,  # 'public', 'internal', 'confidential', 'restricted'
            'type': data_type,  # 'biometric', 'behavioral', 'location', etc.
            'retention_days': retention_days,
            'encryption_required': sensitivity_level in ['confidential', 'restricted'],
            'anonymization_required': sensitivity_level in ['confidential', 'restricted']
        }

    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        encrypted_data = self.cipher.encrypt(data.encode())
        return base64.b64encode(encrypted_data).decode()

    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        encrypted_bytes = base64.b64decode(encrypted_data.encode())
        decrypted_bytes = self.cipher.decrypt(encrypted_bytes)
        return decrypted_bytes.decode()

    def anonymize_data(self, data: Dict[str, Any], fields_to_anonymize: List[str]) -> Dict[str, Any]:
        """Anonymize specific fields in data"""
        anonymized_data = data.copy()

        for field in fields_to_anonymize:
            if field in anonymized_data:
                original_value = anonymized_data[field]
                if isinstance(original_value, str):
                    # Replace with hash
                    hashed = hashlib.sha256(original_value.encode()).hexdigest()
                    anonymized_data[field] = f"ANONYMIZED_{hashed[:8]}"
                elif isinstance(original_value, (int, float)):
                    # Add noise
                    noise = secrets.randbelow(100) - 50  # Random noise between -50 and 50
                    anonymized_data[field] = original_value + noise
                else:
                    # For other types, set to None or generic value
                    anonymized_data[field] = "ANONYMIZED"

        return anonymized_data

    def pseudonymize_data(self, data: Dict[str, Any], fields_to_pseudonymize: List[str]) -> Dict[str, Any]:
        """Pseudonymize specific fields in data"""
        pseudonymized_data = data.copy()

        for field in fields_to_pseudonymize:
            if field in pseudonymized_data:
                original_value = pseudonymized_data[field]
                if isinstance(original_value, str):
                    # Create pseudonym based on hash but keep it consistent
                    pseudonym = f"USER_{hashlib.md5(original_value.encode()).hexdigest()[:8]}"
                    pseudonymized_data[field] = pseudonym

        return pseudonymized_data

    def process_data(self, data_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data according to its classification"""
        if data_id not in self.data_classification:
            raise ValueError(f"Data {data_id} not classified")

        classification = self.data_classification[data_id]
        processed_data = data.copy()

        # Apply appropriate privacy measures
        if classification['anonymization_required']:
            # Determine fields to anonymize based on type
            fields_to_anonymize = []
            if classification['type'] == 'biometric':
                fields_to_anonymize = [k for k in data.keys() if 'id' in k.lower() or 'name' in k.lower()]
            elif classification['type'] == 'location':
                fields_to_anonymize = ['location', 'coordinates', 'address']
            elif classification['type'] == 'behavioral':
                fields_to_anonymize = ['user_id', 'session_id']

            processed_data = self.anonymize_data(processed_data, fields_to_anonymize)

        if classification['encryption_required']:
            # Encrypt the entire data payload
            encrypted_payload = self.encrypt_data(json.dumps(processed_data))
            processed_data = {'encrypted_payload': encrypted_payload, 'format': 'encrypted'}

        # Log the processing
        self.access_logs.append({
            'data_id': data_id,
            'action': 'process',
            'timestamp': time.time(),
            'classification': classification
        })

        return processed_data

    def store_consent(self, user_id: str, purpose: str, granted: bool,
                     expiration: Optional[float] = None):
        """Store user consent for data processing"""
        consent_key = f"{user_id}_{purpose}"
        self.consent_records[consent_key] = {
            'user_id': user_id,
            'purpose': purpose,
            'granted': granted,
            'timestamp': time.time(),
            'expiration': expiration
        }

    def check_consent(self, user_id: str, purpose: str) -> bool:
        """Check if user has consented to data processing for purpose"""
        consent_key = f"{user_id}_{purpose}"
        if consent_key not in self.consent_records:
            return False

        consent = self.consent_records[consent_key]
        if consent['expiration'] and time.time() > consent['expiration']:
            # Consent has expired
            return False

        return consent['granted']

    def get_privacy_report(self) -> Dict:
        """Get privacy compliance report"""
        return {
            'total_classified_data': len(self.data_classification),
            'total_access_logs': len(self.access_logs),
            'total_consent_records': len(self.consent_records),
            'encrypted_data_count': sum(1 for d in self.data_classification.values() if d['encryption_required']),
            'anonymized_data_count': sum(1 for d in self.data_classification.values() if d['anonymization_required'])
        }

class ConsentManager:
    def __init__(self, privacy_manager: DataPrivacyManager):
        """Initialize consent manager"""
        self.privacy_manager = privacy_manager
        self.consent_templates = {}
        self.consent_history = []

    def create_consent_template(self, template_id: str, description: str,
                              data_types: List[str], purposes: List[str]):
        """Create a consent template"""
        self.consent_templates[template_id] = {
            'description': description,
            'data_types': data_types,
            'purposes': purposes,
            'required': True  # Whether this consent is required
        }

    def request_consent(self, user_id: str, template_id: str,
                       custom_purposes: Optional[List[str]] = None) -> Dict:
        """Request consent from user"""
        if template_id not in self.consent_templates:
            raise ValueError(f"Template {template_id} not found")

        template = self.consent_templates[template_id]
        purposes = custom_purposes if custom_purposes else template['purposes']

        consent_request = {
            'user_id': user_id,
            'template_id': template_id,
            'description': template['description'],
            'data_types': template['data_types'],
            'purposes': purposes,
            'timestamp': time.time(),
            'status': 'pending'
        }

        self.consent_history.append(consent_request)
        return consent_request

    def record_consent_response(self, user_id: str, purposes: List[str],
                              granted: bool, expiration_days: int = 365):
        """Record user's consent response"""
        expiration = time.time() + (expiration_days * 24 * 60 * 60) if expiration_days > 0 else None

        for purpose in purposes:
            self.privacy_manager.store_consent(user_id, purpose, granted, expiration)

        response_record = {
            'user_id': user_id,
            'purposes': purposes,
            'granted': granted,
            'timestamp': time.time(),
            'expiration': expiration
        }

        # Update consent history
        for record in reversed(self.consent_history):
            if (record['user_id'] == user_id and
                record['status'] == 'pending' and
                all(p in record['purposes'] for p in purposes)):
                record['status'] = 'granted' if granted else 'denied'
                record['consent_granted'] = granted
                break

        return response_record

    def get_user_consent_status(self, user_id: str) -> Dict:
        """Get consent status for user"""
        user_consents = {}
        for key, record in self.privacy_manager.consent_records.items():
            if record['user_id'] == user_id:
                user_consents[record['purpose']] = {
                    'granted': record['granted'],
                    'timestamp': record['timestamp'],
                    'expires': record['expiration']
                }

        return user_consents

# Example usage
if __name__ == "__main__":
    import time

    print("Testing Data Privacy Framework...")

    # Initialize privacy manager
    privacy_manager = DataPrivacyManager()

    # Classify different types of data
    privacy_manager.classify_data('user_profile_001', 'confidential', 'biometric', retention_days=365)
    privacy_manager.classify_data('location_data_001', 'confidential', 'location', retention_days=30)
    privacy_manager.classify_data('usage_stats_001', 'internal', 'behavioral', retention_days=180)

    # Create sample data
    user_profile = {
        'user_id': 'U12345',
        'name': 'John Doe',
        'age': 35,
        'heart_rate': 72,
        'preferences': {'temperature': 22, 'lighting': 'medium'}
    }

    location_data = {
        'user_id': 'U12345',
        'timestamp': time.time(),
        'coordinates': [40.7128, -74.0060],  # NYC coordinates
        'location_name': 'Times Square'
    }

    # Process data according to classification
    print("Processing user profile...")
    processed_profile = privacy_manager.process_data('user_profile_001', user_profile)
    print(f"Original profile keys: {list(user_profile.keys())}")
    print(f"Processed profile keys: {list(processed_profile.keys())}")

    print("\nProcessing location data...")
    processed_location = privacy_manager.process_data('location_data_001', location_data)
    print(f"Location data processed: {isinstance(processed_location.get('encrypted_payload'), str)}")

    # Test consent management
    print("\nTesting Consent Management...")
    consent_manager = ConsentManager(privacy_manager)

    # Create consent templates
    consent_manager.create_consent_template(
        'data_collection',
        'Consent for collecting usage data to improve services',
        ['behavioral', 'location'],
        ['analytics', 'personalization']
    )

    # Request consent
    consent_request = consent_manager.request_consent('U12345', 'data_collection')
    print(f"Consent requested for: {consent_request['purposes']}")

    # Record consent response
    consent_response = consent_manager.record_consent_response(
        'U12345', ['analytics', 'personalization'], granted=True
    )
    print(f"Consent granted: {consent_response['granted']}")

    # Check consent status
    consent_status = consent_manager.get_user_consent_status('U12345')
    print(f"User consent status: {consent_status}")

    # Get privacy report
    privacy_report = privacy_manager.get_privacy_report()
    print(f"\nPrivacy Report: {privacy_report}")
```

## Practical Exercises

### Exercise 1: Implement a Comprehensive Safety System

**Objective**: Create a complete safety system that integrates multiple safety layers for a robotic application.

**Steps**:
1. Implement sensor-based hazard detection
2. Create emergency stop mechanisms
3. Add safety interlocks and monitoring
4. Integrate with robot control system
5. Test with various safety scenarios

**Expected Outcome**: A multi-layered safety system that can detect hazards, respond appropriately, and ensure safe robot operation.

### Exercise 2: Ethical Decision Framework

**Objective**: Develop an ethical decision-making framework for a specific robotic application.

**Steps**:
1. Identify ethical principles relevant to your application
2. Create decision trees for common ethical dilemmas
3. Implement ethical evaluation algorithms
4. Test with various scenarios
5. Evaluate the framework's effectiveness

**Expected Outcome**: An ethical decision-making system that can guide robot behavior in complex moral situations.

### Exercise 3: Privacy-Preserving Data System

**Objective**: Design a data processing system that maintains user privacy while enabling robot functionality.

**Steps**:
1. Implement data classification and sensitivity levels
2. Add encryption and anonymization capabilities
3. Create consent management system
4. Test with realistic data scenarios
5. Evaluate privacy protection effectiveness

**Expected Outcome**: A privacy-preserving system that can process sensitive data while maintaining user privacy and regulatory compliance.

## Chapter Summary

This chapter covered the critical aspects of safety and ethics in robotics:

1. **Safety Systems**: Implementation of safety-aware control, emergency stops, and risk assessment methodologies to ensure robot operation doesn't harm humans or environment.

2. **Ethical Frameworks**: Development of ethical decision-making systems based on established moral principles and frameworks.

3. **Asimov's Laws**: Implementation approaches for Asimov's Three Laws of Robotics and modern ethical guidelines.

4. **Privacy Protection**: Techniques for preserving user privacy while enabling robotic functionality.

5. **Compliance**: Approaches for ensuring robots comply with safety standards and ethical guidelines.

Safety and ethics are fundamental requirements for the widespread adoption of robotics technology. As robots become more autonomous and integrated into human environments, the importance of robust safety systems and ethical decision-making capabilities becomes increasingly critical.

## Further Reading

1. "Robot Ethics: The Ethical and Social Implications of Robotics" by Lin et al. - Comprehensive ethical framework
2. "Safety Critical Systems Handbook" by Habli - Safety engineering principles
3. "The IEEE Global Initiative on Ethics of Autonomous and Intelligent Systems" - Ethical standards
4. "Privacy-Enhancing Technologies" by Clifton and Juell - Privacy preservation techniques
5. "Engineering a Safer World" by Leveson - Systems safety engineering approaches

## Assessment Questions

1. Design a safety system for a collaborative robot working alongside humans in a manufacturing environment.

2. Implement Asimov's Three Laws in a modern robotic system and discuss the challenges.

3. Analyze the ethical implications of autonomous weapons systems.

4. Design a privacy-preserving data collection system for a healthcare robot.

5. Discuss the trade-offs between robot autonomy and safety in different application domains.

6. Evaluate the effectiveness of current safety standards for service robots.

7. Design an ethical decision-making system for a self-driving car.

8. Analyze the impact of algorithmic bias on robotic systems and society.

9. Discuss the role of transparency and explainability in ethical robotics.

10. Evaluate the regulatory challenges in deploying ethical autonomous systems.

