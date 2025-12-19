---
sidebar_position: 7
title: "Chapter 7: Perception and Computer Vision for Robotics"
---

# Chapter 7: Perception and Computer Vision for Robotics

## Learning Objectives

By the end of this chapter, students will be able to:
- Understand the fundamental principles of robot perception and computer vision
- Implement image processing techniques for robotic applications
- Apply feature detection and matching algorithms for object recognition
- Integrate multiple sensors for robust perception in robotics
- Develop SLAM (Simultaneous Localization and Mapping) algorithms
- Evaluate the performance of perception systems in real-world scenarios
- Implement deep learning approaches for robotic vision tasks
- Design perception pipelines that work effectively in dynamic environments

## Theoretical Foundations

### Introduction to Robot Perception

Robot perception is the process by which robots acquire, interpret, and understand information about their environment. This capability is fundamental to autonomous operation, enabling robots to navigate, manipulate objects, avoid obstacles, and interact with humans. Perception systems typically integrate multiple sensors including cameras, LiDAR, sonar, and tactile sensors to create a comprehensive understanding of the surroundings.

The perception process can be broken down into several key components:
- **Sensing**: Acquisition of raw data from various sensors
- **Preprocessing**: Enhancement and filtering of sensor data
- **Feature Extraction**: Identification of relevant information from sensor data
- **Interpretation**: Understanding the meaning of extracted features
- **Integration**: Combining information from multiple sensors and time steps

Computer vision specifically focuses on interpreting visual information from cameras. This field has seen tremendous advances with the advent of deep learning, enabling robots to recognize objects, understand scenes, and navigate complex environments with increasing accuracy.

### Camera Models and Image Formation

Understanding how cameras capture images is crucial for computer vision applications. The pinhole camera model provides a simplified representation of how 3D points in the world are projected onto a 2D image plane. The relationship between a 3D point (X, Y, Z) in the world coordinate system and its 2D projection (u, v) on the image plane is given by:

u = fx * (X/Z) + cx
v = fy * (Y/Z) + cy

Where (fx, fy) are the focal lengths in pixels, and (cx, cy) is the principal point. This model is extended to include distortion parameters that account for real-world lens imperfections.

Stereo vision extends this concept by using two cameras to estimate depth information, mimicking human binocular vision. By finding corresponding points in the left and right images, triangulation can be used to compute 3D coordinates of scene points.

### Sensor Fusion Fundamentals

Robots rarely rely on a single sensor modality. Sensor fusion combines data from multiple sensors to improve perception accuracy and robustness. Common approaches include:

- **Early fusion**: Combining raw sensor data before processing
- **Feature-level fusion**: Combining extracted features from different sensors
- **Decision-level fusion**: Combining results from individual sensor processing

Kalman filters and particle filters are commonly used for sensor fusion, providing optimal estimates by considering sensor uncertainties and dynamic models.

## Image Processing Techniques

### Basic Image Operations

Let's start with fundamental image processing operations that form the building blocks for more complex perception systems:

```python
#!/usr/bin/env python3

import cv2
import numpy as np
import matplotlib.pyplot as plt

class BasicImageProcessor:
    def __init__(self):
        pass

    def gaussian_blur(self, image, kernel_size=5, sigma=1.0):
        """Apply Gaussian blur to reduce noise"""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    def edge_detection(self, image, low_threshold=50, high_threshold=150):
        """Apply Canny edge detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        return edges

    def morphological_operations(self, image, operation='open', kernel_size=3):
        """Apply morphological operations"""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        if operation == 'open':
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        elif operation == 'close':
            return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        elif operation == 'dilate':
            return cv2.dilate(image, kernel)
        elif operation == 'erode':
            return cv2.erode(image, kernel)
        else:
            return image

    def threshold_image(self, image, threshold_value=127, max_value=255, method='binary'):
        """Apply thresholding to create binary image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        if method == 'binary':
            _, binary = cv2.threshold(gray, threshold_value, max_value, cv2.THRESH_BINARY)
        elif method == 'adaptive':
            binary = cv2.adaptiveThreshold(gray, max_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
        else:
            _, binary = cv2.threshold(gray, threshold_value, max_value, cv2.THRESH_BINARY)

        return binary

    def histogram_equalization(self, image):
        """Apply histogram equalization for contrast enhancement"""
        if len(image.shape) == 3:
            # Convert to YUV for better results
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        else:
            return cv2.equalizeHist(image)

# Example usage
if __name__ == "__main__":
    # Create a sample image for demonstration
    sample_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    processor = BasicImageProcessor()

    # Apply various operations
    blurred = processor.gaussian_blur(sample_image, kernel_size=5, sigma=1.0)
    edges = processor.edge_detection(sample_image)
    thresholded = processor.threshold_image(sample_image, threshold_value=127)

    print("Basic image processing operations completed")
```

### Advanced Filtering Techniques

For more sophisticated image processing, we implement advanced filtering techniques:

```python
#!/usr/bin/env python3

import cv2
import numpy as np
from scipy import ndimage
from scipy.signal import wiener

class AdvancedImageProcessor:
    def __init__(self):
        pass

    def bilateral_filter(self, image, d=9, sigma_color=75, sigma_space=75):
        """Apply bilateral filter for edge-preserving smoothing"""
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    def median_filter(self, image, kernel_size=5):
        """Apply median filter for noise reduction"""
        return cv2.medianBlur(image, kernel_size)

    def anisotropic_diffusion(self, image, num_iter=10, delta_t=0.125, kappa=50):
        """Apply anisotropic diffusion for noise reduction while preserving edges"""
        img = image.astype(np.float64)

        for i in range(num_iter):
            # Calculate gradients
            dx = np.diff(img, axis=1, append=img[:, -1:])
            dy = np.diff(img, axis=0, append=img[-1:, :])

            # Calculate conductance
            c_x = np.exp(-(dx**2) / (kappa**2))
            c_y = np.exp(-(dy**2) / (kappa**2))

            # Update image
            img += delta_t * (
                np.roll(c_x * dx, 1, axis=1) - c_x * dx +
                np.roll(c_y * dy, 1, axis=0) - c_y * dy
            )

        return np.clip(img, 0, 255).astype(np.uint8)

    def wiener_filter(self, image, noise_power=0.01):
        """Apply Wiener filter for noise reduction"""
        if len(image.shape) == 3:
            # Process each channel separately
            filtered = np.zeros_like(image)
            for i in range(image.shape[2]):
                filtered[:, :, i] = wiener(image[:, :, i], noise=noise_power)
            return filtered.astype(np.uint8)
        else:
            return wiener(image, noise=noise_power).astype(np.uint8)

    def unsharp_masking(self, image, strength=1.5, radius=1.0, threshold=0):
        """Apply unsharp masking to enhance image details"""
        # Create blurred version
        blurred = cv2.GaussianBlur(image, (0, 0), radius)

        # Calculate mask
        mask = cv2.subtract(image, blurred)

        # Apply mask to original image
        sharpened = cv2.addWeighted(image, 1.0 + strength, mask, -strength, 0)

        # Apply threshold if specified
        if threshold > 0:
            low_contrast_mask = np.absolute(mask) < threshold
            np.copyto(sharpened, image, where=low_contrast_mask)

        return sharpened

    def morphological_edge_detection(self, image, kernel_size=3):
        """Apply morphological edge detection"""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Top-hat transformation (for bright objects)
        top_hat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

        # Black-hat transformation (for dark objects)
        black_hat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)

        # Combine for enhanced edges
        enhanced = cv2.add(image, top_hat)
        enhanced = cv2.subtract(enhanced, black_hat)

        return enhanced

# Example usage
if __name__ == "__main__":
    # Create a sample image with noise for demonstration
    np.random.seed(42)
    sample_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Add some noise
    noise = np.random.normal(0, 25, sample_image.shape).astype(np.uint8)
    noisy_image = cv2.add(sample_image, noise)

    processor = AdvancedImageProcessor()

    # Apply various advanced operations
    bilateral = processor.bilateral_filter(noisy_image)
    median = processor.median_filter(noisy_image)
    wiener_filtered = processor.wiener_filter(noisy_image)
    sharpened = processor.unsharp_masking(sample_image)

    print("Advanced image processing operations completed")
```

## Feature Detection and Matching

### Corner Detection Algorithms

Feature detection is crucial for identifying distinctive points in images that can be used for tasks like object recognition, tracking, and SLAM:

```python
#!/usr/bin/env python3

import cv2
import numpy as np
import matplotlib.pyplot as plt

class FeatureDetector:
    def __init__(self):
        pass

    def harris_corner_detection(self, image, block_size=2, ksize=3, k=0.04):
        """Harris corner detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gray = np.float32(gray)

        # Calculate Harris corner response
        dst = cv2.cornerHarris(gray, block_size, ksize, k)

        # Dilate the result to mark corners
        dst = cv2.dilate(dst, None)

        # Threshold for an optimal value
        threshold = 0.01 * dst.max()
        corners = np.where(dst > threshold)

        return corners, dst

    def shi_tomasi_detection(self, image, max_corners=100, quality_level=0.01, min_distance=10):
        """Shi-Tomasi corner detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        corners = cv2.goodFeaturesToTrack(
            gray,
            max_corners,
            quality_level,
            min_distance
        )

        return corners

    def fast_corner_detection(self, image, threshold=10, nonmax_suppression=True):
        """FAST corner detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        fast = cv2.FastFeatureDetector_create()
        fast.setThreshold(threshold)
        fast.setNonmaxSuppression(nonmax_suppression)

        keypoints = fast.detect(gray, None)

        return keypoints

    def orb_features(self, image):
        """ORB (Oriented FAST and Rotated BRIEF) features"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(gray, None)

        return keypoints, descriptors

    def sift_features(self, image):
        """SIFT (Scale-Invariant Feature Transform) features"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        try:
            # SIFT is patented, so it might not be available in all OpenCV builds
            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(gray, None)
            return keypoints, descriptors
        except cv2.error:
            print("SIFT not available in this OpenCV build")
            return [], None

    def draw_keypoints(self, image, keypoints, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS):
        """Draw keypoints on image"""
        if isinstance(keypoints, tuple) and len(keypoints) == 2:
            # If keypoints is from detectAndCompute (keypoints, descriptors)
            kp = keypoints[0]
        else:
            kp = keypoints

        return cv2.drawKeypoints(image, kp, None, color=color, flags=flags)

# Example usage
if __name__ == "__main__":
    # Create a sample image for feature detection
    sample_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Add some geometric shapes to create corners
    cv2.rectangle(sample_image, (100, 100), (200, 200), (255, 255, 255), 2)
    cv2.circle(sample_image, (300, 300), 50, (255, 0, 0), 2)
    cv2.line(sample_image, (400, 100), (500, 200), (0, 255, 0), 2)

    detector = FeatureDetector()

    # Detect corners using different methods
    harris_corners, harris_response = detector.harris_corner_detection(sample_image)
    shi_tomasi_corners = detector.shi_tomasi_detection(sample_image)
    fast_keypoints = detector.fast_corner_detection(sample_image)
    orb_keypoints, orb_descriptors = detector.orb_features(sample_image)

    print(f"Harris corners detected: {len(harris_corners[0])}")
    print(f"Shi-Tomasi corners detected: {len(shi_tomasi_corners) if shi_tomasi_corners is not None else 0}")
    print(f"FAST corners detected: {len(fast_keypoints)}")
    print(f"ORB features detected: {len(orb_keypoints) if orb_keypoints is not None else 0}")
```

### Feature Matching Algorithms

Once features are detected, we need to match them across different images or views:

```python
#!/usr/bin/env python3

import cv2
import numpy as np

class FeatureMatcher:
    def __init__(self):
        pass

    def brute_force_match(self, descriptors1, descriptors2, cross_check=True):
        """Brute force feature matching"""
        if descriptors1 is None or descriptors2 is None:
            return []

        bf = cv2.BFMatcher()
        matches = bf.match(descriptors1, descriptors2)

        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        return matches

    def knn_match(self, descriptors1, descriptors2, k=2, ratio_threshold=0.75):
        """K-nearest neighbor feature matching with ratio test"""
        if descriptors1 is None or descriptors2 is None:
            return []

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors1, descriptors2, k=k)

        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)

        return good_matches

    def flann_match(self, descriptors1, descriptors2, ratio_threshold=0.75):
        """FLANN (Fast Library for Approximate Nearest Neighbors) matching"""
        if descriptors1 is None or descriptors2 is None:
            return []

        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)

        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)

        return good_matches

    def draw_matches(self, img1, keypoints1, img2, keypoints2, matches,
                    flags=cv2.DrawMatchesFlags_DEFAULT):
        """Draw feature matches between two images"""
        return cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches,
                              None, flags=flags)

    def estimate_homography(self, keypoints1, keypoints2, matches, reproj_threshold=5.0):
        """Estimate homography matrix from matched keypoints"""
        if len(matches) >= 4:
            # Get matched keypoints
            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Find homography matrix
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reproj_threshold)

            return H, mask
        else:
            return None, None

# Example usage
if __name__ == "__main__":
    # Create two sample images for matching
    img1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Add some similar patterns to both images
    cv2.rectangle(img1, (100, 100), (200, 200), (255, 255, 255), 2)
    cv2.rectangle(img2, (120, 120), (220, 220), (255, 255, 255), 2)

    # Detect features in both images
    detector = FeatureDetector()
    kp1, desc1 = detector.orb_features(img1)
    kp2, desc2 = detector.orb_features(img2)

    # Match features
    matcher = FeatureMatcher()
    matches = matcher.knn_match(desc1, desc2)

    print(f"Found {len(matches)} good matches")

    # Estimate homography if enough matches found
    if len(matches) >= 4:
        H, mask = matcher.estimate_homography(kp1, kp2, matches)
        if H is not None:
            print(f"Homography matrix estimated: {H}")
```

## Deep Learning for Computer Vision

### Convolutional Neural Networks for Object Detection

Deep learning has revolutionized computer vision, particularly for object detection and recognition tasks:

```python
#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import numpy as np

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # Assuming 32x32 input
        self.fc2 = nn.Linear(512, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))  # 32x16x16
        x = self.pool(F.relu(self.conv2(x)))  # 64x8x8
        x = self.pool(F.relu(self.conv3(x)))  # 128x4x4

        # Flatten for fully connected layers
        x = x.view(-1, 128 * 4 * 4)

        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class YOLOv3Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(YOLOv3Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels*2, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels*2)

    def forward(self, x):
        residual = x

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x += residual
        return x

class YOLOv3(nn.Module):
    def __init__(self, num_classes=80):
        super(YOLOv3, self).__init__()
        self.num_classes = num_classes

        # Darknet backbone
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            YOLOv3Block(64, 32),
            YOLOv3Block(64, 64),
            YOLOv3Block(128, 64),
            YOLOv3Block(128, 128),
            YOLOv3Block(256, 128),
            YOLOv3Block(256, 256),
            YOLOv3Block(512, 256),
            YOLOv3Block(512, 512),
            YOLOv3Block(1024, 512),
            YOLOv3Block(1024, 1024),
        ])

        # Detection heads
        self.detection_head = nn.Conv2d(1024, 3*(5 + num_classes), kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Process through residual blocks
        for block in self.res_blocks:
            x = block(x)

        # Detection
        detections = self.detection_head(x)

        return detections

class RoboticVisionPipeline:
    def __init__(self, model_type='simple_cnn'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if model_type == 'simple_cnn':
            self.model = SimpleCNN(num_classes=10)
        elif model_type == 'yolo':
            self.model = YOLOv3(num_classes=80)
        else:
            self.model = SimpleCNN(num_classes=10)

        self.model.to(self.device)

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def preprocess_image(self, image):
        """Preprocess image for neural network"""
        if isinstance(image, np.ndarray):
            # Convert numpy array to tensor
            image = self.transform(image)

        # Add batch dimension
        image = image.unsqueeze(0)
        image = image.to(self.device)

        return image

    def detect_objects(self, image):
        """Detect objects in image using trained model"""
        self.model.eval()

        with torch.no_grad():
            # Preprocess image
            processed_image = self.preprocess_image(image)

            # Forward pass
            outputs = self.model(processed_image)

            # Apply softmax to get probabilities
            probabilities = F.softmax(outputs, dim=1)

            # Get predicted class and confidence
            _, predicted_class = torch.max(probabilities, 1)
            confidence = torch.max(probabilities, 1)[0].item()

            return predicted_class.item(), confidence

    def train_model(self, train_loader, num_epochs=10, learning_rate=0.001):
        """Train the model on provided data"""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.model.train()

        for epoch in range(num_epochs):
            running_loss = 0.0
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(self.device), targets.to(self.device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(data)

                # Calculate loss
                loss = criterion(outputs, targets)

                # Backward pass
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if batch_idx % 100 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}], Loss: {loss.item():.4f}')

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Example usage
if __name__ == "__main__":
    # Create a robotic vision pipeline
    vision_pipeline = RoboticVisionPipeline(model_type='simple_cnn')

    # Create a sample image for testing
    sample_image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)

    # Detect objects in the image
    predicted_class, confidence = vision_pipeline.detect_objects(sample_image)

    print(f"Predicted class: {predicted_class}, Confidence: {confidence:.4f}")
```

### Semantic Segmentation

Semantic segmentation assigns a class label to each pixel in an image, providing detailed scene understanding:

```python
#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegmentationBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=21):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = SegmentationBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = SegmentationBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = SegmentationBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = SegmentationBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = SegmentationBlock(512, 1024)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = SegmentationBlock(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = SegmentationBlock(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = SegmentationBlock(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = SegmentationBlock(128, 64)

        # Output layer
        self.outconv = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.pool1(e1)
        e2 = self.enc2(e2)
        e3 = self.pool2(e2)
        e3 = self.enc3(e3)
        e4 = self.pool3(e3)
        e4 = self.enc4(e4)

        # Bottleneck
        b = self.pool4(e4)
        b = self.bottleneck(b)

        # Decoder
        d4 = self.upconv4(b)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)

        # Output
        out = self.outconv(d1)
        return out

class SegmentationPipeline:
    def __init__(self, num_classes=21):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UNet(num_classes=num_classes)
        self.model.to(self.device)

    def segment_image(self, image):
        """Segment image and return class probabilities for each pixel"""
        self.model.eval()

        with torch.no_grad():
            # Convert image to tensor and normalize
            if isinstance(image, np.ndarray):
                # Convert from HWC to CHW format
                image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
                # Normalize to [0, 1] range
                image_tensor = image_tensor / 255.0
                # Add batch dimension
                image_tensor = image_tensor.unsqueeze(0)

            image_tensor = image_tensor.to(self.device)

            # Forward pass
            outputs = self.model(image_tensor)

            # Apply softmax to get probabilities
            probabilities = F.softmax(outputs, dim=1)

            # Get predicted class for each pixel
            predicted_classes = torch.argmax(probabilities, dim=1)

            return predicted_classes.cpu().numpy()[0]  # Remove batch dimension

    def get_segmentation_mask(self, image, class_id):
        """Get binary mask for a specific class"""
        segmentation = self.segment_image(image)
        mask = (segmentation == class_id).astype(np.uint8) * 255
        return mask

# Example usage
if __name__ == "__main__":
    # Create a segmentation pipeline
    seg_pipeline = SegmentationPipeline(num_classes=21)

    # Create a sample image for segmentation
    sample_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    # Perform segmentation
    segmentation = seg_pipeline.segment_image(sample_image)

    print(f"Segmentation completed. Shape: {segmentation.shape}")
    print(f"Unique classes found: {np.unique(segmentation)}")
```

## Sensor Fusion for Robotic Perception

### Kalman Filter Implementation

Kalman filters are essential for fusing data from multiple sensors with different noise characteristics:

```python
#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, state_dim, measurement_dim):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim

        # State vector [x, y, vx, vy] for 2D tracking
        self.x = np.zeros((state_dim, 1))

        # State covariance matrix
        self.P = np.eye(state_dim)

        # Process noise covariance
        self.Q = np.eye(state_dim) * 0.1

        # Measurement noise covariance
        self.R = np.eye(measurement_dim) * 1.0

        # State transition matrix (constant velocity model)
        self.F = np.eye(state_dim)

        # Measurement matrix
        self.H = np.zeros((measurement_dim, state_dim))

        # Control matrix (not used in this example)
        self.B = None

    def predict(self, dt=1.0):
        """Predict next state"""
        # Update state transition matrix for constant velocity model
        self.F[0, 2] = dt  # x = x + vx*dt
        self.F[1, 3] = dt  # y = y + vy*dt

        # Predict state
        self.x = np.dot(self.F, self.x)

        # Predict covariance
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, measurement):
        """Update state with measurement"""
        # Calculate innovation (measurement residual)
        y = measurement - np.dot(self.H, self.x)

        # Calculate innovation covariance
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R

        # Calculate Kalman gain
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # Update state
        self.x = self.x + np.dot(K, y)

        # Update covariance
        I = np.eye(self.P.shape[0])
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

class ExtendedKalmanFilter:
    def __init__(self, state_dim, measurement_dim):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim

        # State vector
        self.x = np.zeros((state_dim, 1))

        # State covariance matrix
        self.P = np.eye(state_dim)

        # Process noise covariance
        self.Q = np.eye(state_dim) * 0.1

        # Measurement noise covariance
        self.R = np.eye(measurement_dim) * 1.0

    def predict(self, dt=1.0):
        """Predict next state using nonlinear model"""
        # For this example, we'll use a simple nonlinear model
        # Update state based on nonlinear dynamics
        # This is a placeholder - actual implementation depends on the system
        self.x[0] = self.x[0] + dt * self.x[2]  # x = x + vx*dt
        self.x[1] = self.x[1] + dt * self.x[3]  # y = y + vy*dt

        # Linearize the system around current state to get F matrix
        F = self.get_jacobian_F(dt)

        # Predict covariance
        self.P = np.dot(np.dot(F, self.P), F.T) + self.Q

    def get_jacobian_F(self, dt):
        """Get Jacobian of the motion model"""
        F = np.eye(self.state_dim)
        F[0, 2] = dt  # dx/dvx
        F[1, 3] = dt  # dy/dvy
        return F

    def update(self, measurement):
        """Update state with measurement using nonlinear measurement model"""
        # Predicted measurement using nonlinear model
        h_x = self.measurement_function()

        # Innovation
        y = measurement - h_x

        # Jacobian of measurement function
        H = self.get_jacobian_H()

        # Innovation covariance
        S = np.dot(np.dot(H, self.P), H.T) + self.R

        # Kalman gain
        K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))

        # Update state
        self.x = self.x + np.dot(K, y)

        # Update covariance
        I = np.eye(self.P.shape[0])
        self.P = np.dot((I - np.dot(K, H)), self.P)

    def measurement_function(self):
        """Nonlinear measurement function"""
        # For example, if we measure range and bearing
        range_val = np.sqrt(self.x[0]**2 + self.x[1]**2)
        bearing = np.arctan2(self.x[1], self.x[0])
        return np.array([[range_val], [bearing]])

    def get_jacobian_H(self):
        """Jacobian of measurement function"""
        # Calculate partial derivatives of measurement function
        x, y = self.x[0, 0], self.x[1, 0]
        r = np.sqrt(x**2 + y**2)

        H = np.zeros((self.measurement_dim, self.state_dim))
        if r > 0:
            H[0, 0] = x / r  # dr/dx
            H[0, 1] = y / r  # dr/dy
            H[1, 0] = -y / r**2  # db/dx
            H[1, 1] = x / r**2  # db/dy

        return H

class MultiSensorFusion:
    def __init__(self):
        # Initialize Kalman filter for 2D position tracking
        self.kf = KalmanFilter(state_dim=4, measurement_dim=2)  # [x, y, vx, vy] with [x, y] measurements

        # Set measurement matrix to extract position from state
        self.kf.H = np.array([[1, 0, 0, 0],    # Measure x position
                             [0, 1, 0, 0]])    # Measure y position

    def process_camera_measurement(self, x, y, timestamp):
        """Process camera measurement"""
        measurement = np.array([[x], [y]])
        self.kf.update(measurement)

    def process_lidar_measurement(self, x, y, timestamp):
        """Process LiDAR measurement"""
        # LiDAR might have different noise characteristics
        original_R = self.kf.R.copy()
        self.kf.R = np.eye(2) * 0.5  # Lower noise for LiDAR
        measurement = np.array([[x], [y]])
        self.kf.update(measurement)
        self.kf.R = original_R  # Restore original noise

    def process_imu_prediction(self, ax, ay, dt):
        """Process IMU prediction to update state"""
        # Update acceleration-based predictions
        self.kf.x[2] += ax * dt  # Update velocity x
        self.kf.x[3] += ay * dt  # Update velocity y

    def get_current_state(self):
        """Get current estimated state [x, y, vx, vy]"""
        return self.kf.x.flatten()

# Example usage
if __name__ == "__main__":
    # Create multi-sensor fusion system
    fusion_system = MultiSensorFusion()

    # Simulate sensor measurements
    true_positions = []
    estimated_positions = []
    measurements = []

    # Initial state
    fusion_system.kf.x = np.array([[0.0], [0.0], [1.0], [0.5]])  # [x, y, vx, vy]

    # Simulate tracking for 100 time steps
    for t in range(100):
        dt = 0.1

        # Predict next state
        fusion_system.kf.predict(dt)

        # Simulate measurements with noise
        true_x = fusion_system.kf.x[0, 0] + np.random.normal(0, 0.1)
        true_y = fusion_system.kf.x[1, 0] + np.random.normal(0, 0.1)

        # Process camera measurement
        fusion_system.process_camera_measurement(true_x, true_y, t*dt)

        # Store for visualization
        true_positions.append([true_x, true_y])
        estimated_positions.append([fusion_system.kf.x[0, 0], fusion_system.kf.x[1, 0]])
        measurements.append([true_x, true_y])

    print("Multi-sensor fusion simulation completed")
    print(f"Final estimated position: ({fusion_system.kf.x[0, 0]:.2f}, {fusion_system.kf.x[1, 0]:.2f})")
```

## SLAM (Simultaneous Localization and Mapping)

### Visual SLAM Implementation

```python
#!/usr/bin/env python3

import numpy as np
import cv2
from collections import deque
import matplotlib.pyplot as plt

class Frame:
    def __init__(self, image, pose=np.eye(4)):
        self.image = image
        self.pose = pose  # 4x4 transformation matrix
        self.keypoints = None
        self.descriptors = None
        self.id = None

class MapPoint:
    def __init__(self, position, descriptor):
        self.position = position  # 3D position in world coordinates
        self.descriptor = descriptor  # Feature descriptor
        self.observations = []  # List of (frame_id, keypoint_idx) tuples
        self.id = None

class VisualSLAM:
    def __init__(self):
        self.frames = []
        self.map_points = []
        self.current_frame_id = 0
        self.current_map_point_id = 0

        # Feature detector and matcher
        self.detector = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Camera parameters (example values)
        self.fx = 500.0
        self.fy = 500.0
        self.cx = 320.0
        self.cy = 240.0
        self.camera_matrix = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])

        # Current pose
        self.current_pose = np.eye(4)

        # Essential matrix decomposition
        self.ransac_threshold = 1.0

    def process_frame(self, image):
        """Process a new frame and update map"""
        # Create frame object
        frame = Frame(image.copy())
        frame.id = self.current_frame_id

        # Extract features
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)

        if descriptors is None:
            return False

        frame.keypoints = keypoints
        frame.descriptors = descriptors

        # Initialize with first frame
        if len(self.frames) == 0:
            self.frames.append(frame)
            self.current_frame_id += 1
            return True

        # Match features with previous frame
        prev_frame = self.frames[-1]
        matches = self.matcher.knnMatch(prev_frame.descriptors, descriptors, k=2)

        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        # Require minimum number of matches
        if len(good_matches) < 20:
            self.frames.append(frame)
            self.current_frame_id += 1
            return False

        # Extract matched points
        prev_pts = np.float32([prev_frame.keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        curr_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Estimate essential matrix
        E, mask = cv2.findEssentialMat(
            prev_pts, curr_pts,
            self.camera_matrix,
            method=cv2.RANSAC,
            threshold=self.ransac_threshold
        )

        if E is None:
            self.frames.append(frame)
            self.current_frame_id += 1
            return False

        # Recover pose
        _, R, t, _ = cv2.recoverPose(E, prev_pts, curr_pts, self.camera_matrix)

        # Create transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()

        # Update current pose
        self.current_pose = self.current_pose @ np.linalg.inv(T)
        frame.pose = self.current_pose.copy()

        # Add frame to map
        self.frames.append(frame)
        self.current_frame_id += 1

        # Update map points (simplified)
        self.update_map_points(prev_pts, curr_pts, good_matches, len(self.frames)-2, len(self.frames)-1)

        return True

    def update_map_points(self, prev_pts, curr_pts, matches, prev_frame_idx, curr_frame_idx):
        """Update map points with new observations"""
        # Convert 2D points to 3D rays and triangulate
        prev_frame = self.frames[prev_frame_idx]
        curr_frame = self.frames[curr_frame_idx]

        # Get relative pose between frames
        relative_pose = np.linalg.inv(prev_frame.pose) @ curr_frame.pose

        # Triangulate points
        R1 = prev_frame.pose[:3, :3]
        t1 = prev_frame.pose[:3, 3]
        R2 = curr_frame.pose[:3, :3]
        t2 = curr_frame.pose[:3, 3]

        # For simplicity, just add some points to the map
        for i, match in enumerate(matches):
            if i < min(5, len(matches)):  # Add first 5 matches as map points
                # Create a 3D point (simplified triangulation)
                pt_3d = self.triangulate_point(
                    prev_pts[i].flatten(),
                    curr_pts[i].flatten(),
                    R1, t1, R2, t2
                )

                if pt_3d is not None:
                    # Create map point
                    map_point = MapPoint(pt_3d, curr_frame.descriptors[match.trainIdx])
                    map_point.id = self.current_map_point_id
                    self.current_map_point_id += 1

                    # Add observations
                    map_point.observations.append((prev_frame_idx, match.queryIdx))
                    map_point.observations.append((curr_frame_idx, match.trainIdx))

                    self.map_points.append(map_point)

    def triangulate_point(self, pt1, pt2, R1, t1, R2, t2):
        """Triangulate a 3D point from two camera views"""
        # Convert 2D points to normalized coordinates
        pt1_norm = np.array([(pt1[0] - self.cx) / self.fx, (pt1[1] - self.cy) / self.fy, 1.0])
        pt2_norm = np.array([(pt2[0] - self.cx) / self.fx, (pt2[1] - self.cy) / self.fy, 1.0])

        # Form the equation system for triangulation
        # [R1|t1] and [R2|t2] are the camera projection matrices
        P1 = np.hstack([R1, t1.reshape(3, 1)])
        P2 = np.hstack([R2, t2.reshape(3, 1)])

        # Create matrix A for linear triangulation
        A = np.zeros((4, 4))
        A[0] = pt1_norm[0] * P1[2, :] - P1[0, :]
        A[1] = pt1_norm[1] * P1[2, :] - P1[1, :]
        A[2] = pt2_norm[0] * P2[2, :] - P2[0, :]
        A[3] = pt2_norm[1] * P2[2, :] - P2[1, :]

        # Solve using SVD
        _, _, V = np.linalg.svd(A)
        X = V[-1, :4]  # Take the last row of V

        if X[3] != 0:
            X = X / X[3]  # Normalize
            return X[:3]  # Return 3D coordinates
        else:
            return None

    def get_current_pose(self):
        """Get current camera pose"""
        return self.current_pose

    def get_tracked_points(self):
        """Get all map points"""
        return self.map_points

# Example usage
if __name__ == "__main__":
    # Create a simple synthetic sequence for testing
    slam = VisualSLAM()

    # Generate synthetic frames (in practice, these would come from a camera)
    for i in range(10):
        # Create a synthetic image with some features
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        # Add some random features (circles)
        for j in range(20):
            center = (np.random.randint(50, 600), np.random.randint(50, 400))
            cv2.circle(image, center, 5, (255, 255, 255), -1)

        # Process the frame
        success = slam.process_frame(image)

        if success:
            print(f"Frame {i} processed successfully")
            pose = slam.get_current_pose()
            print(f"Current pose:\n{pose[:3, 3]}")
        else:
            print(f"Frame {i} failed to process")

    print(f"SLAM completed with {len(slam.get_tracked_points())} map points")
```

## Practical Exercises

### Exercise 1: Implement a Real-time Object Detection System

**Objective**: Create a real-time object detection system that can identify and track objects in a video stream.

**Steps**:
1. Implement a basic CNN for object classification
2. Add bounding box detection capabilities
3. Create a tracking system to follow objects across frames
4. Integrate with ROS2 for robotic applications
5. Test the system with various objects and lighting conditions

**Expected Outcome**: A functional object detection and tracking system that can run in real-time on video input.

### Exercise 2: Multi-Sensor Fusion for Robot Navigation

**Objective**: Implement a sensor fusion system that combines camera, LiDAR, and IMU data for robust robot localization.

**Steps**:
1. Set up Kalman filter for state estimation
2. Integrate camera pose estimation
3. Add LiDAR distance measurements
4. Include IMU acceleration data
5. Test the system in a simulated environment

**Expected Outcome**: A robust localization system that provides accurate position estimates by combining multiple sensor modalities.

### Exercise 3: SLAM System Implementation

**Objective**: Implement a complete visual SLAM system for environment mapping and robot localization.

**Steps**:
1. Create feature detection and matching pipeline
2. Implement pose estimation from feature correspondences
3. Develop map management system
4. Add loop closure detection
5. Test with real or simulated camera data

**Expected Outcome**: A working SLAM system that can simultaneously localize the robot and build a map of the environment.

## Chapter Summary

This chapter covered the essential concepts and techniques for robot perception and computer vision:

1. **Fundamentals**: Understanding camera models, image formation, and sensor fusion principles that form the basis of robotic perception.

2. **Image Processing**: Techniques for enhancing, filtering, and analyzing visual data to extract meaningful information.

3. **Feature Detection**: Methods for identifying distinctive points in images that can be used for object recognition, tracking, and mapping.

4. **Deep Learning**: Modern approaches using neural networks for object detection, segmentation, and scene understanding.

5. **Sensor Fusion**: Techniques for combining data from multiple sensors to improve perception accuracy and robustness.

6. **SLAM**: Simultaneous Localization and Mapping techniques that enable robots to navigate unknown environments while building maps.

The integration of these perception techniques enables robots to understand and interact with their environment effectively. Modern robotic systems increasingly rely on deep learning approaches combined with traditional computer vision techniques to achieve robust performance in real-world scenarios.

## Further Reading

1. "Computer Vision: Algorithms and Applications" by Richard Szeliski - Comprehensive coverage of computer vision techniques
2. "Probabilistic Robotics" by Thrun, Burgard, and Fox - Essential text on robot perception and state estimation
3. "Multiple View Geometry in Computer Vision" by Hartley and Zisserman - Mathematical foundations of 3D vision
4. "Learning OpenCV 3" by Bradski and Kaehler - Practical guide to OpenCV implementation
5. "Deep Learning for Computer Vision" by Rajalingappaa and Shanmuganathan - Modern deep learning approaches

## Assessment Questions

1. Explain the pinhole camera model and derive the equations for 3D to 2D projection.

2. Compare different feature detection algorithms (Harris, SIFT, ORB) in terms of computational complexity and performance.

3. Describe the mathematical formulation of the Kalman filter and explain how it handles uncertainty in sensor measurements.

4. Implement a stereo vision system to estimate depth from two camera images.

5. Discuss the challenges and solutions in visual SLAM, including feature matching, pose estimation, and map management.

6. Analyze the advantages and disadvantages of different sensor fusion approaches for robotic perception.

7. Explain how convolutional neural networks can be used for object detection and segmentation in robotic applications.

8. Describe the process of triangulation in 3D reconstruction and its applications in robotics.

9. Compare classical computer vision approaches with deep learning methods for robotic perception tasks.

10. Design a complete perception pipeline for a mobile robot operating in dynamic environments.

