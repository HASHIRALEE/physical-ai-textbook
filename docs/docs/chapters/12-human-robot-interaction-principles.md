---
sidebar_position: 12
title: "Chapter 12: Human-Robot Interaction Principles"
---

# Chapter 12: Human-Robot Interaction Principles

## Learning Objectives

By the end of this chapter, students will be able to:
- Understand the fundamental principles of human-robot interaction (HRI)
- Apply psychological and social theories to design effective HRI systems
- Implement various interaction modalities including speech, gesture, and touch
- Design user interfaces that facilitate natural and intuitive robot communication
- Evaluate HRI systems using established metrics and methodologies
- Address safety and ethical considerations in human-robot interactions
- Develop adaptive systems that learn from human behavior and preferences
- Analyze the impact of cultural and social factors on HRI design

## Theoretical Foundations

### Introduction to Human-Robot Interaction

Human-Robot Interaction (HRI) is an interdisciplinary field that focuses on the design, development, and evaluation of robotic systems for human use. Unlike traditional human-computer interaction (HCI), HRI deals with embodied agents that exist in physical space and can move, manipulate objects, and interact with the environment in complex ways.

The fundamental challenges in HRI include:

**Social Acceptance**: Robots must be designed to be acceptable to humans in various social contexts. This involves understanding cultural norms, social expectations, and human comfort levels with robotic agents.

**Natural Communication**: Humans naturally communicate through multiple modalities including speech, gestures, facial expressions, and body language. Effective HRI systems must support these natural communication channels.

**Trust and Reliability**: Humans must trust robots to perform tasks reliably and safely. This trust is built through consistent behavior, transparent decision-making, and appropriate responses to unexpected situations.

**Social Intelligence**: Robots must understand and respond appropriately to social cues, context, and human emotions to interact effectively in social environments.

### Psychological and Social Theories in HRI

Several psychological and social theories inform the design of HRI systems:

**Social Presence Theory**: This theory suggests that humans respond to robots as social actors when they exhibit certain social cues. The more a robot exhibits social presence, the more humans will engage with it socially.

**Media Equation Theory**: This theory posits that people treat computers and other media as real social actors. This extends to robots, where humans may apply social rules and expectations to robotic agents.

**Uncanny Valley Hypothesis**: Proposed by Masahiro Mori, this hypothesis suggests that human observers have positive responses to robots that are either clearly artificial or very human-like, but negative responses to robots that appear almost human but not quite.

**Social Cognitive Theory**: This theory emphasizes the importance of observational learning, imitation, and modeling in human behavior. In HRI, this suggests that robots can influence human behavior through their actions and social cues.

### Models of Human-Robot Interaction

Several models help frame the understanding of HRI:

**Collaborative Model**: In this model, humans and robots work together as teammates, each contributing their respective strengths to accomplish tasks. This model emphasizes shared decision-making and mutual adaptation.

**Complementary Model**: This model focuses on the complementary nature of human and robot capabilities, where robots handle tasks they are better at (repetitive, precise, dangerous) while humans handle tasks requiring creativity, judgment, and adaptability.

**Supervisory Model**: In this model, humans supervise and guide robot behavior, making high-level decisions while robots handle low-level execution. This model is common in teleoperation and semi-autonomous systems.

**Companion Model**: This model treats robots as social companions that provide emotional support, companionship, and social interaction. This model is particularly relevant for assistive and therapeutic robotics.

## Interaction Modalities

### Speech and Natural Language Processing

Speech is one of the most natural and intuitive forms of human communication. Implementing effective speech interaction in robots requires several components:

```python
#!/usr/bin/env python3

import speech_recognition as sr
import pyttsx3
import nltk
import spacy
from typing import Dict, List, Tuple, Optional
import re
import json

class SpeechRecognitionSystem:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Initialize text-to-speech engine
        self.tts_engine = pyttsx3.init()

        # Set TTS properties
        self.tts_engine.setProperty('rate', 150)  # Speed of speech
        self.tts_engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)

        # Initialize NLP components
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("SpaCy English model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None

        # Intent recognition patterns
        self.intent_patterns = {
            'greeting': [
                r'hello', r'hi', r'hey', r'good morning', r'good afternoon', r'good evening'
            ],
            'navigation': [
                r'go to', r'move to', r'go to the', r'go to', r'navigate to'
            ],
            'manipulation': [
                r'pick up', r'grasp', r'grab', r'lift', r'put down', r'place', r'move'
            ],
            'information': [
                r'what is', r'tell me about', r'how do', r'can you', r'what can you'
            ],
            'confirmation': [
                r'yes', r'no', r'correct', r'right', r'wrong', r'okay', r'stop'
            ]
        }

    def listen(self) -> Optional[str]:
        """Listen for speech and return recognized text"""
        with self.microphone as source:
            print("Listening...")
            # Adjust for ambient noise
            self.recognizer.adjust_for_ambient_noise(source)
            try:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                text = self.recognizer.recognize_google(audio)
                print(f"Recognized: {text}")
                return text.lower()
            except sr.WaitTimeoutError:
                print("Timeout: No speech detected")
                return None
            except sr.UnknownValueError:
                print("Could not understand audio")
                return None
            except sr.RequestError as e:
                print(f"Error with speech recognition service: {e}")
                return None

    def speak(self, text: str):
        """Convert text to speech"""
        print(f"Robot says: {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """Extract named entities from text using NLP"""
        if not self.nlp:
            return []

        doc = self.nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities

    def recognize_intent(self, text: str) -> str:
        """Recognize the intent from the given text"""
        text_lower = text.lower()

        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return intent

        return 'unknown'

    def parse_command(self, text: str) -> Dict:
        """Parse a command and extract relevant information"""
        intent = self.recognize_intent(text)
        entities = self.extract_entities(text)

        command = {
            'intent': intent,
            'entities': entities,
            'raw_text': text
        }

        # Extract specific information based on intent
        if intent == 'navigation':
            # Look for location entities
            locations = [ent[0] for ent in entities if ent[1] in ['LOC', 'GPE', 'FAC']]
            if locations:
                command['target_location'] = locations[0]

        elif intent == 'manipulation':
            # Look for object entities
            objects = [ent[0] for ent in entities if ent[1] in ['PRODUCT', 'OBJECT']]
            if objects:
                command['target_object'] = objects[0]

        return command

class DialogueManager:
    def __init__(self):
        self.conversation_history = []
        self.current_context = {}
        self.user_preferences = {}

        # Initialize speech system
        self.speech_system = SpeechRecognitionSystem()

    def process_input(self, user_input: str) -> str:
        """Process user input and generate appropriate response"""
        # Parse the command
        parsed_command = self.speech_system.parse_command(user_input)

        # Store in conversation history
        self.conversation_history.append({
            'user_input': user_input,
            'parsed_command': parsed_command,
            'timestamp': time.time()
        })

        # Generate response based on intent
        response = self.generate_response(parsed_command)

        # Update context
        self.update_context(parsed_command, response)

        return response

    def generate_response(self, parsed_command: Dict) -> str:
        """Generate response based on parsed command"""
        intent = parsed_command['intent']

        responses = {
            'greeting': [
                "Hello! How can I assist you today?",
                "Hi there! What can I do for you?",
                "Good to see you! How may I help?"
            ],
            'navigation': [
                f"I can help you navigate to {parsed_command.get('target_location', 'that location')}.",
                f"Moving towards {parsed_command.get('target_location', 'the target')}.",
                "I understand you want me to go somewhere."
            ],
            'manipulation': [
                f"I can help with that {parsed_command.get('target_object', 'object')}.",
                "I'll assist you with manipulating objects.",
                "Object manipulation is one of my capabilities."
            ],
            'information': [
                "I can provide information about various topics.",
                "I'm here to answer your questions.",
                "I can help you learn more about things."
            ],
            'confirmation': [
                "I understand your response.",
                "Got it, thank you for confirming.",
                "Acknowledged."
            ],
            'unknown': [
                "I'm not sure I understand. Could you please rephrase?",
                "I didn't catch that. Can you say it again?",
                "I'm still learning. Could you explain differently?"
            ]
        }

        import random
        return random.choice(responses.get(intent, responses['unknown']))

    def update_context(self, parsed_command: Dict, response: str):
        """Update conversation context"""
        # Update based on entities mentioned
        for entity, label in parsed_command.get('entities', []):
            if label in ['PERSON']:
                self.current_context['person'] = entity
            elif label in ['GPE', 'LOC']:
                self.current_context['location'] = entity
            elif label in ['OBJECT', 'PRODUCT']:
                self.current_context['object'] = entity

# Example usage
if __name__ == "__main__":
    import time

    # Initialize dialogue manager
    dm = DialogueManager()

    # Simulate conversation
    sample_inputs = [
        "Hello robot",
        "Can you go to the kitchen?",
        "Please pick up the red cup",
        "What is your name?"
    ]

    for user_input in sample_inputs:
        print(f"User: {user_input}")
        response = dm.process_input(user_input)
        print(f"Robot: {response}")
        print(f"Context: {dm.current_context}")
        print("---")
        time.sleep(1)  # Simulate processing time
```

### Gesture Recognition and Interpretation

Gesture recognition enables robots to understand human body language and non-verbal communication:

```python
#!/usr/bin/env python3

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Dict, Optional
import math

class GestureRecognitionSystem:
    def __init__(self):
        # Initialize MediaPipe components
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        # Initialize gesture recognizers
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.7
        )

        # Define gesture patterns
        self.gesture_patterns = {
            'wave': self.is_wave_gesture,
            'point': self.is_pointing_gesture,
            'stop': self.is_stop_gesture,
            'come_here': self.is_come_here_gesture,
            'thumbs_up': self.is_thumbs_up_gesture,
            'peace_sign': self.is_peace_sign_gesture
        }

    def process_hand_landmarks(self, image: np.ndarray) -> Dict:
        """Process hand landmarks from image"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)

        gesture_data = {
            'hands': [],
            'gesture_type': None,
            'gesture_confidence': 0.0
        }

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract hand landmarks
                hand_points = []
                for landmark in hand_landmarks.landmark:
                    h, w, c = image.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    hand_points.append((cx, cy))

                gesture_data['hands'].append(hand_points)

                # Check for known gestures
                for gesture_name, gesture_func in self.gesture_patterns.items():
                    if gesture_func(hand_points):
                        gesture_data['gesture_type'] = gesture_name
                        gesture_data['gesture_confidence'] = 0.9  # High confidence for rule-based detection
                        break

        return gesture_data

    def process_body_pose(self, image: np.ndarray) -> Dict:
        """Process body pose landmarks from image"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)

        pose_data = {
            'pose_landmarks': None,
            'body_orientation': None,
            'gaze_direction': None
        }

        if results.pose_landmarks:
            pose_data['pose_landmarks'] = results.pose_landmarks

            # Calculate body orientation (simplified)
            landmarks = results.pose_landmarks.landmark
            nose = landmarks[mp.solutions.pose.PoseLandmark.NOSE.value]
            left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]

            # Calculate angle to determine orientation
            shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
            shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2

            dx = nose.x - shoulder_center_x
            dy = nose.y - shoulder_center_y

            angle = math.atan2(dy, dx)
            pose_data['body_orientation'] = angle

        return pose_data

    def is_wave_gesture(self, hand_points: List[Tuple[int, int]]) -> bool:
        """Detect waving gesture"""
        if len(hand_points) < 21:
            return False

        # Check if index finger is extended while others are folded
        # Simplified detection - in practice, you'd use more sophisticated analysis
        wrist = hand_points[0]
        index_tip = hand_points[8]
        middle_tip = hand_points[12]
        ring_tip = hand_points[16]
        pinky_tip = hand_points[20]
        thumb_tip = hand_points[4]

        # Check if index finger is extended (higher position = lower y value)
        return (index_tip[1] < wrist[1] and
                middle_tip[1] > wrist[1] and
                ring_tip[1] > wrist[1] and
                pinky_tip[1] > wrist[1])

    def is_pointing_gesture(self, hand_points: List[Tuple[int, int]]) -> bool:
        """Detect pointing gesture"""
        if len(hand_points) < 21:
            return False

        # Check if index finger is extended while others are folded
        wrist = hand_points[0]
        index_tip = hand_points[8]
        middle_tip = hand_points[12]
        ring_tip = hand_points[16]
        pinky_tip = hand_points[20]

        return (index_tip[1] < middle_tip[1] and
                index_tip[1] < ring_tip[1] and
                index_tip[1] < pinky_tip[1])

    def is_stop_gesture(self, hand_points: List[Tuple[int, int]]) -> bool:
        """Detect stop gesture (palm facing forward)"""
        if len(hand_points) < 21:
            return False

        # Check if all fingers are extended (palm open)
        wrist = hand_points[0]
        index_tip = hand_points[8]
        middle_tip = hand_points[12]
        ring_tip = hand_points[16]
        pinky_tip = hand_points[20]

        # All fingertips should be at similar height (palm open)
        avg_finger_height = (index_tip[1] + middle_tip[1] + ring_tip[1] + pinky_tip[1]) / 4
        height_variance = max(abs(index_tip[1] - avg_finger_height),
                             abs(middle_tip[1] - avg_finger_height),
                             abs(ring_tip[1] - avg_finger_height),
                             abs(pinky_tip[1] - avg_finger_height))

        return height_variance < 30  # Threshold for "flat" palm

    def is_come_here_gesture(self, hand_points: List[Tuple[int, int]]) -> bool:
        """Detect come-here gesture (index finger extended, other fingers folded)"""
        return self.is_pointing_gesture(hand_points)

    def is_thumbs_up_gesture(self, hand_points: List[Tuple[int, int]]) -> bool:
        """Detect thumbs up gesture"""
        if len(hand_points) < 21:
            return False

        # Check if thumb is extended upward while others are folded
        thumb_tip = hand_points[4]
        index_tip = hand_points[8]
        middle_tip = hand_points[12]
        ring_tip = hand_points[16]
        pinky_tip = hand_points[20]

        return (thumb_tip[1] < index_tip[1] and  # Thumb higher than other fingers
                thumb_tip[1] < middle_tip[1] and
                thumb_tip[1] < ring_tip[1] and
                thumb_tip[1] < pinky_tip[1])

    def is_peace_sign_gesture(self, hand_points: List[Tuple[int, int]]) -> bool:
        """Detect peace sign (index and middle fingers extended)"""
        if len(hand_points) < 21:
            return False

        # Check if index and middle fingers are extended while others are folded
        index_tip = hand_points[8]
        middle_tip = hand_points[12]
        ring_tip = hand_points[16]
        pinky_tip = hand_points[20]
        thumb_tip = hand_points[4]

        return (index_tip[1] < ring_tip[1] and  # Index higher than ring
                middle_tip[1] < ring_tip[1] and  # Middle higher than ring
                middle_tip[1] < pinky_tip[1] and  # Middle higher than pinky
                thumb_tip[1] > index_tip[1])  # Thumb lower than index (folded)

    def process_image(self, image: np.ndarray) -> Dict:
        """Process image for both hand gestures and body pose"""
        hand_data = self.process_hand_landmarks(image)
        pose_data = self.process_body_pose(image)

        return {
            'hand_gestures': hand_data,
            'body_pose': pose_data
        }

class GestureInterpreter:
    def __init__(self):
        self.gesture_system = GestureRecognitionSystem()
        self.gesture_history = []
        self.interaction_context = {}

    def interpret_gesture(self, gesture_data: Dict) -> Dict:
        """Interpret gesture and generate appropriate response"""
        hand_gestures = gesture_data['hand_gestures']
        body_pose = gesture_data['body_pose']

        interpretation = {
            'detected_gestures': [],
            'body_language': {},
            'interaction_intent': None,
            'confidence': 0.0
        }

        # Process hand gestures
        if hand_gestures['gesture_type']:
            interpretation['detected_gestures'].append({
                'type': hand_gestures['gesture_type'],
                'confidence': hand_gestures['gesture_confidence']
            })

            # Map gestures to interaction intents
            gesture_intent_map = {
                'wave': 'greeting',
                'stop': 'attention_stop',
                'come_here': 'approach',
                'point': 'direct_attention',
                'thumbs_up': 'approval',
                'peace_sign': 'positive_feedback'
            }

            interpretation['interaction_intent'] = gesture_intent_map.get(
                hand_gestures['gesture_type'], 'unknown'
            )
            interpretation['confidence'] = hand_gestures['gesture_confidence']

        # Process body language
        if body_pose['body_orientation'] is not None:
            interpretation['body_language']['orientation'] = body_pose['body_orientation']

        # Add to history
        self.gesture_history.append(interpretation.copy())
        if len(self.gesture_history) > 10:  # Keep last 10 gestures
            self.gesture_history.pop(0)

        return interpretation

# Example usage
if __name__ == "__main__":
    # Initialize gesture interpreter
    interpreter = GestureInterpreter()

    # Simulate gesture processing (in practice, this would process camera frames)
    print("Gesture recognition system initialized")
    print("Available gestures:", list(interpreter.gesture_system.gesture_patterns.keys()))
```

### Visual and Haptic Feedback Systems

```python
#!/usr/bin/env python3

import numpy as np
import pygame
import time
from typing import Tuple, List, Dict
import threading
import queue

class VisualFeedbackSystem:
    def __init__(self, width: int = 800, height: int = 600):
        """Initialize visual feedback system"""
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Robot Visual Feedback System")
        self.clock = pygame.time.Clock()

        # Colors
        self.colors = {
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'yellow': (255, 255, 0),
            'purple': (255, 0, 255),
            'cyan': (0, 255, 255)
        }

        # State variables
        self.running = True
        self.feedback_elements = []
        self.text_elements = []

        # Start display thread
        self.display_thread = threading.Thread(target=self._display_loop)
        self.display_thread.daemon = True
        self.display_thread.start()

    def add_feedback_element(self, element_type: str, position: Tuple[int, int],
                           size: Tuple[int, int], color: str = 'white',
                           duration: float = 2.0):
        """Add a visual feedback element"""
        element = {
            'type': element_type,
            'position': position,
            'size': size,
            'color': self.colors.get(color, self.colors['white']),
            'duration': duration,
            'start_time': time.time()
        }
        self.feedback_elements.append(element)

    def add_text_feedback(self, text: str, position: Tuple[int, int],
                         color: str = 'white', size: int = 24):
        """Add text feedback"""
        font = pygame.font.Font(None, size)
        text_surface = font.render(text, True, self.colors.get(color, self.colors['white']))
        text_element = {
            'surface': text_surface,
            'position': position,
            'text': text
        }
        self.text_elements.append(text_element)

    def clear_feedback(self):
        """Clear all feedback elements"""
        self.feedback_elements.clear()
        self.text_elements.clear()

    def _display_loop(self):
        """Main display loop"""
        while self.running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            # Clear screen
            self.screen.fill(self.colors['black'])

            # Draw feedback elements
            current_time = time.time()
            elements_to_remove = []

            for i, element in enumerate(self.feedback_elements):
                if current_time - element['start_time'] > element['duration']:
                    elements_to_remove.append(i)
                    continue

                if element['type'] == 'rectangle':
                    pygame.draw.rect(
                        self.screen,
                        element['color'],
                        (*element['position'], *element['size'])
                    )
                elif element['type'] == 'circle':
                    pygame.draw.circle(
                        self.screen,
                        element['color'],
                        element['position'],
                        element['size'][0]  # radius
                    )
                elif element['type'] == 'pulse':
                    # Create pulsing effect based on time
                    pulse_factor = (np.sin(current_time * 5) + 1) / 2  # 0-1 oscillation
                    pulse_size = tuple(int(s * (0.8 + 0.4 * pulse_factor)) for s in element['size'])
                    pygame.draw.rect(
                        self.screen,
                        element['color'],
                        (*element['position'], *pulse_size)
                    )

            # Remove expired elements
            for i in reversed(elements_to_remove):
                del self.feedback_elements[i]

            # Draw text elements
            for text_element in self.text_elements:
                self.screen.blit(text_element['surface'], text_element['position'])

            # Update display
            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS

    def stop(self):
        """Stop the visual feedback system"""
        self.running = False
        if self.display_thread.is_alive():
            self.display_thread.join()

class HapticFeedbackSystem:
    def __init__(self):
        """Initialize haptic feedback system (simulated)"""
        self.vibration_queue = queue.Queue()
        self.running = True

        # Start haptic feedback thread
        self.haptic_thread = threading.Thread(target=self._haptic_loop)
        self.haptic_thread.daemon = True
        self.haptic_thread.start()

    def send_vibration(self, intensity: float, duration: float, pattern: str = 'continuous'):
        """Send vibration command"""
        if 0 <= intensity <= 1 and duration > 0:
            command = {
                'intensity': intensity,
                'duration': duration,
                'pattern': pattern,
                'timestamp': time.time()
            }
            self.vibration_queue.put(command)

    def send_force_feedback(self, force_vector: Tuple[float, float, float], duration: float):
        """Send force feedback command (simulated)"""
        command = {
            'type': 'force',
            'force_vector': force_vector,
            'duration': duration,
            'timestamp': time.time()
        }
        self.vibration_queue.put(command)

    def _haptic_loop(self):
        """Main haptic feedback loop"""
        while self.running:
            try:
                # Get next command from queue
                command = self.vibration_queue.get(timeout=0.1)

                # Simulate haptic feedback
                if command['type'] == 'force':
                    print(f"Force feedback: {command['force_vector']} for {command['duration']}s")
                else:
                    print(f"Vibration: intensity={command['intensity']}, "
                          f"duration={command['duration']}s, pattern={command['pattern']}")

                # Simulate the haptic effect
                time.sleep(command['duration'])

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in haptic loop: {e}")

    def stop(self):
        """Stop the haptic feedback system"""
        self.running = False
        if self.haptic_thread.is_alive():
            self.haptic_thread.join()

class MultimodalFeedbackManager:
    def __init__(self):
        """Manage multiple feedback modalities"""
        self.visual_system = VisualFeedbackSystem()
        self.haptic_system = HapticFeedbackSystem()

        # Feedback mapping
        self.feedback_map = {
            'success': self._feedback_success,
            'error': self._feedback_error,
            'warning': self._feedback_warning,
            'attention': self._feedback_attention,
            'confirmation': self._feedback_confirmation
        }

    def provide_feedback(self, feedback_type: str, message: str = "",
                        visual_only: bool = False, haptic_only: bool = False):
        """Provide multimodal feedback based on type"""
        if feedback_type in self.feedback_map:
            self.feedback_map[feedback_type](message, visual_only, haptic_only)
        else:
            print(f"Unknown feedback type: {feedback_type}")

    def _feedback_success(self, message: str, visual_only: bool, haptic_only: bool):
        """Provide success feedback"""
        if not haptic_only:
            self.visual_system.add_feedback_element(
                'pulse', (100, 100), (50, 50), 'green', 1.0
            )
            self.visual_system.add_text_feedback(
                message or "Success!", (160, 110), 'green', 32
            )

        if not visual_only:
            self.haptic_system.send_vibration(0.3, 0.2, 'short_burst')

    def _feedback_error(self, message: str, visual_only: bool, haptic_only: bool):
        """Provide error feedback"""
        if not haptic_only:
            self.visual_system.add_feedback_element(
                'pulse', (100, 100), (50, 50), 'red', 1.5
            )
            self.visual_system.add_text_feedback(
                message or "Error!", (160, 110), 'red', 32
            )

        if not visual_only:
            self.haptic_system.send_vibration(0.8, 0.5, 'long_pulse')

    def _feedback_warning(self, message: str, visual_only: bool, haptic_only: bool):
        """Provide warning feedback"""
        if not haptic_only:
            self.visual_system.add_feedback_element(
                'pulse', (100, 100), (50, 50), 'yellow', 1.0
            )
            self.visual_system.add_text_feedback(
                message or "Warning!", (160, 110), 'yellow', 32
            )

        if not visual_only:
            self.haptic_system.send_vibration(0.5, 0.3, 'double_burst')

    def _feedback_attention(self, message: str, visual_only: bool, haptic_only: bool):
        """Provide attention feedback"""
        if not haptic_only:
            # Flashing effect for attention
            for i in range(3):
                self.visual_system.add_feedback_element(
                    'pulse', (100, 100), (50, 50), 'blue', 0.5
                )
                time.sleep(0.3)

        if not visual_only:
            self.haptic_system.send_vibration(0.4, 0.4, 'triple_burst')

    def _feedback_confirmation(self, message: str, visual_only: bool, haptic_only: bool):
        """Provide confirmation feedback"""
        if not haptic_only:
            self.visual_system.add_feedback_element(
                'circle', (100, 100), (30, 30), 'cyan', 0.8
            )
            self.visual_system.add_text_feedback(
                message or "Confirmed", (140, 105), 'cyan', 28
            )

        if not visual_only:
            self.haptic_system.send_vibration(0.2, 0.1, 'short_burst')

    def clear_all_feedback(self):
        """Clear all feedback elements"""
        self.visual_system.clear_feedback()

    def stop(self):
        """Stop all feedback systems"""
        self.visual_system.stop()
        self.haptic_system.stop()

# Example usage
if __name__ == "__main__":
    import time

    # Initialize feedback manager
    feedback_manager = MultimodalFeedbackManager()

    print("Testing multimodal feedback system...")

    # Test different feedback types
    test_feedbacks = [
        ('success', 'Task completed successfully!'),
        ('error', 'Error occurred!'),
        ('warning', 'Warning: Check parameters'),
        ('attention', 'Attention required'),
        ('confirmation', 'Action confirmed')
    ]

    for feedback_type, message in test_feedbacks:
        print(f"Providing {feedback_type} feedback: {message}")
        feedback_manager.provide_feedback(feedback_type, message)
        time.sleep(2)

    # Test custom feedback
    feedback_manager.provide_feedback('success', 'Custom success message', visual_only=True)
    time.sleep(2)

    feedback_manager.clear_all_feedback()
    print("Feedback test completed")

    # Stop systems
    feedback_manager.stop()
```

## User Interface Design for HRI

### Social Interface Principles

Creating effective interfaces for human-robot interaction requires understanding both technical and social aspects:

```python
#!/usr/bin/env python3

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from typing import Dict, List, Callable, Any
import json

class SocialRobotInterface:
    def __init__(self):
        """Initialize the social robot interface"""
        self.root = tk.Tk()
        self.root.title("Social Robot Interface")
        self.root.geometry("800x600")

        # Interface components
        self.interface_elements = {}
        self.user_preferences = {}
        self.interaction_history = []

        # Initialize interface
        self._setup_interface()

        # Robot state
        self.robot_state = {
            'name': 'Robbie',
            'status': 'idle',
            'battery': 85,
            'current_task': 'standby',
            'social_mode': 'friendly'
        }

        # Start interface update thread
        self.interface_running = True
        self.update_thread = threading.Thread(target=self._interface_update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()

    def _setup_interface(self):
        """Setup the main interface components"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Robot status panel
        status_frame = ttk.LabelFrame(main_frame, text="Robot Status", padding="10")
        status_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        # Status labels
        self.status_labels = {}
        status_items = [
            ('name', 'Name:'),
            ('status', 'Status:'),
            ('battery', 'Battery:'),
            ('task', 'Current Task:')
        ]

        for i, (key, label) in enumerate(status_items):
            ttk.Label(status_frame, text=label).grid(row=0, column=i*2, sticky=tk.W, padx=5)
            self.status_labels[key] = ttk.Label(status_frame, text="")
            self.status_labels[key].grid(row=0, column=i*2+1, sticky=tk.W, padx=5)

        # Interaction panel
        interaction_frame = ttk.LabelFrame(main_frame, text="Interaction", padding="10")
        interaction_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        # Command input
        ttk.Label(interaction_frame, text="Command:").grid(row=0, column=0, sticky=tk.W)
        self.command_entry = ttk.Entry(interaction_frame, width=50)
        self.command_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        self.command_entry.bind('<Return>', self._send_command)

        send_btn = ttk.Button(interaction_frame, text="Send", command=self._send_command)
        send_btn.grid(row=0, column=2, padx=5)

        # Quick commands
        quick_commands_frame = ttk.Frame(interaction_frame)
        quick_commands_frame.grid(row=1, column=0, columnspan=3, pady=10)

        quick_commands = [
            ("Greet", lambda: self._send_text_command("Hello")),
            ("Follow", lambda: self._send_text_command("Please follow me")),
            ("Stop", lambda: self._send_text_command("Stop")),
            ("Help", lambda: self._send_text_command("Can you help me?")),
            ("Wait", lambda: self._send_text_command("Wait here"))
        ]

        for i, (text, command) in enumerate(quick_commands):
            btn = ttk.Button(quick_commands_frame, text=text, command=command)
            btn.grid(row=0, column=i, padx=2)

        # Chat display
        chat_frame = ttk.LabelFrame(main_frame, text="Conversation", padding="10")
        chat_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5, padx=(5, 0))

        # Chat text widget
        self.chat_text = tk.Text(chat_frame, height=15, width=40)
        scrollbar = ttk.Scrollbar(chat_frame, orient="vertical", command=self.chat_text.yview)
        self.chat_text.configure(yscrollcommand=scrollbar.set)

        self.chat_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # User preferences panel
        prefs_frame = ttk.LabelFrame(main_frame, text="Preferences", padding="10")
        prefs_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        # Social mode selection
        ttk.Label(prefs_frame, text="Social Mode:").grid(row=0, column=0, sticky=tk.W)
        self.social_mode_var = tk.StringVar(value="friendly")
        modes = ["friendly", "professional", "enthusiastic", "calm"]
        for i, mode in enumerate(modes):
            rb = ttk.Radiobutton(prefs_frame, text=mode.title(), variable=self.social_mode_var,
                               value=mode, command=self._update_social_mode)
            rb.grid(row=0, column=i+1, sticky=tk.W, padx=5)

        # Personality settings
        ttk.Label(prefs_frame, text="Personality:").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))

        self.personality_vars = {
            'extroversion': tk.DoubleVar(value=5.0),
            'agreeableness': tk.DoubleVar(value=7.0),
            'conscientiousness': tk.DoubleVar(value=6.0)
        }

        for i, (trait, var) in enumerate(self.personality_vars.items()):
            ttk.Label(prefs_frame, text=f"{trait.title()}:").grid(row=2+i, column=0, sticky=tk.W)
            scale = ttk.Scale(prefs_frame, from_=1, to=10, variable=var, orient='horizontal')
            scale.grid(row=2+i, column=1, sticky=(tk.W, tk.E), padx=5)
            value_label = ttk.Label(prefs_frame, text="5.0")
            value_label.grid(row=2+i, column=2, sticky=tk.W)

            # Update value label when scale changes
            def update_label(var, label, trait):
                def callback(*args):
                    value = var.get()
                    label.config(text=f"{value:.1f}")
                    self.user_preferences[f'personality_{trait}'] = value
                return callback

            var.trace('w', update_label(var, value_label, trait))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)

    def _send_command(self, event=None):
        """Send command to robot"""
        command = self.command_entry.get().strip()
        if command:
            self._add_to_conversation(f"You: {command}", "user")
            self.command_entry.delete(0, tk.END)

            # Simulate robot response
            self._simulate_robot_response(command)

    def _send_text_command(self, command: str):
        """Send a predefined text command"""
        self.command_entry.delete(0, tk.END)
        self.command_entry.insert(0, command)
        self._send_command()

    def _simulate_robot_response(self, command: str):
        """Simulate robot response to command"""
        import random

        responses = {
            'greeting': [
                "Hello! How can I assist you today?",
                "Hi there! What can I do for you?",
                "Good to see you! How may I help?"
            ],
            'navigation': [
                "I can help you navigate. Where would you like to go?",
                "I'm ready to guide you. What's your destination?",
                "I can escort you there safely."
            ],
            'help': [
                "I'm here to help. What do you need assistance with?",
                "How can I be of service?",
                "I'm ready to assist you with that."
            ],
            'follow': [
                "I'll follow you now. Please proceed.",
                "Following you. Maintaining safe distance.",
                "I'm right behind you."
            ],
            'stop': [
                "Stopping now.",
                "Halted. How can I help?",
                "I've stopped. What's next?"
            ],
            'default': [
                "I understand you said: '{command}'. How can I help?",
                "Thanks for letting me know: '{command}'.",
                "I've processed your request: '{command}'."
            ]
        }

        # Determine response type based on command
        command_lower = command.lower()
        if any(word in command_lower for word in ['hello', 'hi', 'hey', 'greet']):
            response_type = 'greeting'
        elif any(word in command_lower for word in ['go', 'navigate', 'move', 'guide', 'escort']):
            response_type = 'navigation'
        elif any(word in command_lower for word in ['help', 'assist', 'support']):
            response_type = 'help'
        elif any(word in command_lower for word in ['follow', 'come', 'behind']):
            response_type = 'follow'
        elif any(word in command_lower for word in ['stop', 'halt', 'wait']):
            response_type = 'stop'
        else:
            response_type = 'default'

        # Get appropriate response
        if response_type == 'default':
            response = random.choice(responses['default']).format(command=command)
        else:
            response = random.choice(responses[response_type])

        # Add to conversation after delay to simulate processing
        self.root.after(1000, lambda: self._add_to_conversation(f"Robbie: {response}", "robot"))

    def _add_to_conversation(self, message: str, sender: str):
        """Add message to conversation display"""
        self.chat_text.config(state=tk.NORMAL)
        self.chat_text.insert(tk.END, message + "\n")
        self.chat_text.config(state=tk.DISABLED)
        self.chat_text.see(tk.END)  # Auto-scroll to end

        # Add to history
        self.interaction_history.append({
            'message': message,
            'sender': sender,
            'timestamp': time.time()
        })

        if len(self.interaction_history) > 100:  # Keep last 100 messages
            self.interaction_history.pop(0)

    def _update_social_mode(self):
        """Update robot's social mode"""
        new_mode = self.social_mode_var.get()
        self.robot_state['social_mode'] = new_mode
        self._add_to_conversation(f"Social mode changed to: {new_mode}", "system")

    def _interface_update_loop(self):
        """Update interface elements periodically"""
        while self.interface_running:
            # Update status labels
            for key, label_widget in self.status_labels.items():
                if key == 'name':
                    label_widget.config(text=self.robot_state['name'])
                elif key == 'status':
                    label_widget.config(text=self.robot_state['status'])
                elif key == 'battery':
                    label_widget.config(text=f"{self.robot_state['battery']}%")
                elif key == 'task':
                    label_widget.config(text=self.robot_state['current_task'])

            time.sleep(0.1)  # Update every 100ms

    def run(self):
        """Run the interface"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.interface_running = False

    def stop(self):
        """Stop the interface"""
        self.interface_running = False
        self.root.quit()

# Example usage
if __name__ == "__main__":
    # Create and run the social robot interface
    interface = SocialRobotInterface()

    print("Social Robot Interface started. You can interact with the robot through the GUI.")
    print("Available quick commands: Greet, Follow, Stop, Help, Wait")
    print("Press Ctrl+C to exit.")

    try:
        interface.run()
    except KeyboardInterrupt:
        print("\nShutting down interface...")
        interface.stop()
```

## Safety and Ethical Considerations

### Safety Frameworks in HRI

```python
#!/usr/bin/env python3

import threading
import time
from typing import Dict, List, Tuple, Callable
import logging
from enum import Enum

class SafetyLevel(Enum):
    SAFE = 0
    WARNING = 1
    DANGER = 2
    EMERGENCY = 3

class SafetyZone:
    def __init__(self, name: str, min_distance: float, max_distance: float,
                 safety_level: SafetyLevel):
        self.name = name
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.safety_level = safety_level

class SafetyManager:
    def __init__(self):
        """Initialize safety management system"""
        self.safety_zones = [
            SafetyZone("Personal Space", 0.0, 1.0, SafetyLevel.DANGER),
            SafetyZone("Interaction Zone", 1.0, 2.0, SafetyLevel.WARNING),
            SafetyZone("Social Zone", 2.0, 4.0, SafetyLevel.SAFE),
            SafetyZone("Public Zone", 4.0, float('inf'), SafetyLevel.SAFE)
        ]

        self.current_safety_level = SafetyLevel.SAFE
        self.emergency_stop = False
        self.safety_violations = []
        self.safety_callbacks = []

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Start safety monitoring thread
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._safety_monitor_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

    def add_safety_callback(self, callback: Callable[[SafetyLevel, str], None]):
        """Add callback for safety events"""
        self.safety_callbacks.append(callback)

    def check_safety(self, distances: Dict[str, float]) -> Dict[str, SafetyLevel]:
        """Check safety for multiple detected objects"""
        safety_status = {}

        for obj_name, distance in distances.items():
            # Determine safety zone
            zone_level = SafetyLevel.SAFE
            for zone in self.safety_zones:
                if zone.min_distance <= distance < zone.max_distance:
                    zone_level = zone.safety_level
                    break

            safety_status[obj_name] = zone_level

            # Log safety events
            if zone_level != SafetyLevel.SAFE:
                self.logger.warning(f"Safety event: {obj_name} at {distance:.2f}m in {zone_level.name} zone")
                self.safety_violations.append({
                    'object': obj_name,
                    'distance': distance,
                    'zone': zone.name,
                    'level': zone_level,
                    'timestamp': time.time()
                })

                # Trigger callbacks
                for callback in self.safety_callbacks:
                    callback(zone_level, f"{obj_name} at {distance:.2f}m")

        # Update current safety level (highest level of concern)
        if safety_status:
            self.current_safety_level = max(safety_status.values())

        return safety_status

    def trigger_emergency_stop(self, reason: str = "Safety violation"):
        """Trigger emergency stop"""
        self.emergency_stop = True
        self.logger.critical(f"EMERGENCY STOP: {reason}")

        # Call all safety callbacks
        for callback in self.safety_callbacks:
            callback(SafetyLevel.EMERGENCY, reason)

    def reset_emergency_stop(self):
        """Reset emergency stop"""
        self.emergency_stop = False
        self.logger.info("Emergency stop reset")

    def _safety_monitor_loop(self):
        """Continuous safety monitoring loop"""
        while self.monitoring_active:
            # In a real system, this would continuously check sensor data
            # For simulation, we'll just sleep
            time.sleep(0.1)

    def get_safety_report(self) -> Dict:
        """Get current safety status report"""
        return {
            'current_level': self.current_safety_level.name,
            'emergency_stop': self.emergency_stop,
            'violations_count': len(self.safety_violations),
            'recent_violations': self.safety_violations[-10:]  # Last 10 violations
        }

    def stop_monitoring(self):
        """Stop safety monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread.is_alive():
            self.monitoring_thread.join()

class EthicalDecisionMaker:
    def __init__(self):
        """Initialize ethical decision making system"""
        self.ethical_principles = {
            'beneficence': 0.8,      # Do good
            'non_malfeasance': 0.9,  # Do no harm
            'autonomy': 0.7,         # Respect human autonomy
            'justice': 0.6,          # Fair treatment
            'veracity': 0.9          # Truthfulness
        }

        self.decision_history = []
        self.current_context = {}

    def evaluate_action(self, action: str, context: Dict) -> Dict:
        """Evaluate an action based on ethical principles"""
        self.current_context = context

        # Calculate ethical scores for each principle
        ethical_scores = {}

        for principle, weight in self.ethical_principles.items():
            score = self._calculate_principle_score(principle, action, context)
            ethical_scores[principle] = {
                'score': score,
                'weight': weight,
                'weighted_score': score * weight
            }

        # Calculate overall ethical score
        total_weighted_score = sum(item['weighted_score'] for item in ethical_scores.values())
        total_weights = sum(item['weight'] for item in ethical_scores.values())
        overall_score = total_weighted_score / total_weights if total_weights > 0 else 0

        decision = {
            'action': action,
            'ethical_scores': ethical_scores,
            'overall_score': overall_score,
            'recommendation': self._get_recommendation(overall_score),
            'timestamp': time.time()
        }

        self.decision_history.append(decision)
        if len(self.decision_history) > 100:  # Keep last 100 decisions
            self.decision_history.pop(0)

        return decision

    def _calculate_principle_score(self, principle: str, action: str, context: Dict) -> float:
        """Calculate score for a specific ethical principle"""
        # This is a simplified model - in practice, this would be much more complex
        scores = {
            'beneficence': self._score_beneficence(action, context),
            'non_malfeasance': self._score_non_malfeasance(action, context),
            'autonomy': self._score_autonomy(action, context),
            'justice': self._score_justice(action, context),
            'veracity': self._score_veracity(action, context)
        }

        return scores.get(principle, 0.5)  # Default to neutral score

    def _score_beneficence(self, action: str, context: Dict) -> float:
        """Score how much the action promotes well-being"""
        positive_actions = ['help', 'assist', 'support', 'guide', 'protect']
        negative_actions = ['ignore', 'harm', 'obstruct']

        if any(pos in action.lower() for pos in positive_actions):
            return 0.9
        elif any(neg in action.lower() for neg in negative_actions):
            return 0.1
        return 0.5

    def _score_non_malfeasance(self, action: str, context: Dict) -> float:
        """Score how much the action avoids harm"""
        harmful_actions = ['harm', 'injure', 'damage', 'threaten', 'attack']
        safe_actions = ['avoid', 'protect', 'warn', 'stop']

        if any(harm in action.lower() for harm in harmful_actions):
            return 0.1
        elif any(safe in action.lower() for safe in safe_actions):
            return 0.9
        return 0.8  # Default to high safety score

    def _score_autonomy(self, action: str, context: Dict) -> float:
        """Score how much the action respects human autonomy"""
        autonomy_respecting = ['ask', 'request', 'offer', 'suggest', 'allow']
        autonomy_violating = ['force', 'compel', 'require', 'demand']

        if any(respect in action.lower() for respect in autonomy_respecting):
            return 0.9
        elif any(violate in action.lower() for violate in autonomy_violating):
            return 0.1
        return 0.6

    def _score_justice(self, action: str, context: Dict) -> float:
        """Score how fair the action is"""
        # Simplified - in practice, this would consider context like user demographics, etc.
        fair_actions = ['equal', 'fair', 'same', 'consistent']
        unfair_actions = ['prefer', 'discriminate', 'exclude']

        if any(fair in action.lower() for fair in fair_actions):
            return 0.8
        elif any(unfair in action.lower() for unfair in unfair_actions):
            return 0.2
        return 0.7

    def _score_veracity(self, action: str, context: Dict) -> float:
        """Score truthfulness of the action"""
        truth_actions = ['inform', 'explain', 'clarify', 'admit']
        deceptive_actions = ['deceive', 'mislead', 'hide', 'lie']

        if any(truth in action.lower() for truth in truth_actions):
            return 0.9
        elif any(deceive in action.lower() for deceive in deceptive_actions):
            return 0.1
        return 0.8

    def _get_recommendation(self, overall_score: float) -> str:
        """Get recommendation based on overall score"""
        if overall_score >= 0.8:
            return "Proceed"
        elif overall_score >= 0.6:
            return "Proceed with caution"
        elif overall_score >= 0.4:
            return "Consider alternatives"
        else:
            return "Do not proceed"

    def get_ethics_report(self) -> Dict:
        """Get ethics decision report"""
        if not self.decision_history:
            return {'message': 'No ethical decisions made yet'}

        recent_decisions = self.decision_history[-10:]  # Last 10 decisions
        avg_score = sum(d['overall_score'] for d in recent_decisions) / len(recent_decisions)

        return {
            'total_decisions': len(self.decision_history),
            'recent_decisions': recent_decisions,
            'average_ethical_score': avg_score,
            'decision_trend': 'improving' if avg_score > 0.7 else 'needs_attention'
        }

class HRIController:
    def __init__(self):
        """Initialize HRI controller with safety and ethics"""
        self.safety_manager = SafetyManager()
        self.ethics_manager = EthicalDecisionMaker()

        # Register safety callback
        self.safety_manager.add_safety_callback(self._on_safety_event)

        # Robot state
        self.robot_state = {
            'position': (0, 0, 0),
            'velocity': (0, 0, 0),
            'orientation': 0,
            'status': 'idle'
        }

    def _on_safety_event(self, level: SafetyLevel, message: str):
        """Handle safety events"""
        print(f"SAFETY EVENT: {level.name} - {message}")

        # Take appropriate action based on safety level
        if level == SafetyLevel.EMERGENCY:
            self.emergency_stop()
        elif level == SafetyLevel.DANGER:
            self.slow_down()
        elif level == SafetyLevel.WARNING:
            self.warn_user()

    def evaluate_robot_action(self, action: str, context: Dict) -> Dict:
        """Evaluate a robot action for safety and ethics"""
        # Check safety first
        safety_status = self.safety_manager.get_safety_report()

        if safety_status['emergency_stop']:
            return {
                'safe_to_proceed': False,
                'safety_issue': True,
                'ethics_evaluation': None,
                'reason': 'Emergency stop active'
            }

        # Evaluate ethics
        ethics_evaluation = self.ethics_manager.evaluate_action(action, context)

        # Combine safety and ethics
        safe_to_proceed = (
            safety_status['current_level'] in [SafetyLevel.SAFE, SafetyLevel.WARNING] and
            ethics_evaluation['recommendation'] in ['Proceed', 'Proceed with caution']
        )

        return {
            'safe_to_proceed': safe_to_proceed,
            'safety_status': safety_status,
            'ethics_evaluation': ethics_evaluation,
            'final_recommendation': 'PROCEED' if safe_to_proceed else 'HOLD'
        }

    def emergency_stop(self):
        """Execute emergency stop"""
        print("EMERGENCY STOP: Halting all robot motion")
        # In a real system, this would send stop commands to all actuators

    def slow_down(self):
        """Slow down robot motion"""
        print("SLOW DOWN: Reducing robot speed for safety")
        # In a real system, this would reduce velocity commands

    def warn_user(self):
        """Warn user about potential safety issue"""
        print("WARNING: Potential safety concern detected")
        # In a real system, this might trigger audio/visual alerts

    def get_system_status(self) -> Dict:
        """Get overall system status"""
        return {
            'safety': self.safety_manager.get_safety_report(),
            'ethics': self.ethics_manager.get_ethics_report(),
            'robot_state': self.robot_state
        }

# Example usage
if __name__ == "__main__":
    # Initialize HRI controller
    controller = HRIController()

    # Test scenarios
    test_scenarios = [
        {
            'action': 'approach human',
            'context': {'distance': 0.5, 'human_behavior': 'calm'}
        },
        {
            'action': 'help elderly person',
            'context': {'age_group': 'elderly', 'need': 'assistance'}
        },
        {
            'action': 'navigate through crowd',
            'context': {'crowd_density': 'high', 'space_constraints': 'narrow'}
        }
    ]

    print("Testing HRI safety and ethics evaluation...")

    for i, scenario in enumerate(test_scenarios):
        print(f"\nScenario {i+1}: {scenario['action']}")
        result = controller.evaluate_robot_action(scenario['action'], scenario['context'])

        print(f"Safe to proceed: {result['safe_to_proceed']}")
        print(f"Ethics recommendation: {result['ethics_evaluation']['recommendation']}")
        print(f"Overall score: {result['ethics_evaluation']['overall_score']:.2f}")

    # Get system status
    status = controller.get_system_status()
    print(f"\nSystem Status:")
    print(f"  Safety Level: {status['safety']['current_level']}")
    print(f"  Ethics Trend: {status['ethics']['decision_trend']}")

    # Cleanup
    controller.safety_manager.stop_monitoring()
    print("\nHRI safety and ethics system test completed")
```

## Practical Exercises

### Exercise 1: Implement a Multimodal Interaction System

**Objective**: Create a complete HRI system that integrates speech, gesture, and visual feedback.

**Steps**:
1. Implement speech recognition and synthesis
2. Add gesture recognition capabilities
3. Create visual feedback displays
4. Integrate all modalities into a cohesive system
5. Test with real users and evaluate effectiveness

**Expected Outcome**: A functional multimodal HRI system that can respond to various forms of human input.

### Exercise 2: Design an Ethical Decision Framework

**Objective**: Develop an ethical decision-making system for a social robot.

**Steps**:
1. Identify relevant ethical principles for your application
2. Create evaluation criteria for different types of decisions
3. Implement the decision framework
4. Test with various scenarios
5. Evaluate the system's performance and fairness

**Expected Outcome**: An ethical decision-making system that can guide robot behavior in complex situations.

### Exercise 3: Safety-Critical HRI System

**Objective**: Design a safety-aware HRI system for physical interaction.

**Steps**:
1. Define safety requirements for your application
2. Implement safety monitoring and response systems
3. Create emergency stop mechanisms
4. Test safety responses under various conditions
5. Validate the system's safety performance

**Expected Outcome**: A safety-aware HRI system that can protect humans during physical interaction.

## Chapter Summary

This chapter covered the essential principles of human-robot interaction:

1. **Theoretical Foundations**: Understanding psychological and social theories that inform HRI design, including social presence, media equation, and uncanny valley concepts.

2. **Interaction Modalities**: Implementing various communication channels including speech recognition, gesture interpretation, and multimodal feedback systems.

3. **User Interface Design**: Creating social interfaces that facilitate natural and intuitive interaction between humans and robots.

4. **Safety Considerations**: Implementing safety frameworks and monitoring systems to protect humans during interaction.

5. **Ethical Frameworks**: Developing ethical decision-making systems that guide robot behavior in complex social situations.

Effective HRI requires careful consideration of human factors, social norms, and ethical principles. Successful HRI systems must balance technical capabilities with human-centered design to create positive and productive interactions.

## Further Reading

1. "Human-Robot Interaction: A Survey" by Goodrich and Schultz - Comprehensive overview of HRI research
2. "The Oxford Handbook of Social Robotics" by Sharkey and Capozzi - Social aspects of robotics
3. "Designing Socially Embedded Robots" by Feil-Seifer and Matarić - Socially assistive robotics
4. "Robot Ethics: The Ethical and Social Implications of Robotics" by Lin et al. - Ethical considerations
5. "Human-Robot Interaction in Assistive Robotics" by Belpaeme et al. - Applications in assistive technology

## Assessment Questions

1. Explain the key psychological theories that influence human-robot interaction design.

2. Design a multimodal interaction system that integrates speech, gesture, and visual feedback.

3. Analyze the uncanny valley hypothesis and its implications for robot design.

4. Implement a safety monitoring system for human-robot physical interaction.

5. Discuss the ethical considerations in social robotics and propose a decision-making framework.

6. Evaluate the effectiveness of different interaction modalities for various user groups.

7. Design a social interface for a specific HRI application (e.g., eldercare, education, service).

8. Compare different models of human-robot interaction (collaborative, supervisory, companion).

9. Implement an ethical decision-making system for a social robot.

10. Assess the cultural and social factors that influence HRI acceptance and effectiveness.

