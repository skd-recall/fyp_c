"""
Frame Stabilizer for Pose Detection
Implements moving average smoothing to reduce jitter and false detections
"""

from collections import deque
from typing import Dict, Optional, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FrameStabilizer:
    """
    Stabilize measurements using moving average
    Reduces jitter from frame-to-frame variations
    """
    
    def __init__(self, window_size: int = 3):
        """
        Initialize frame stabilizer
        
        Args:
            window_size: Number of frames to average (default: 3)
        """
        self.window_size = window_size
        self.angle_buffers: Dict[str, deque] = {}
        self.landmark_buffers: Dict[str, deque] = {}
        
        logger.info(f"Frame stabilizer initialized with window size: {window_size}")
    
    def add_angle_measurement(self, angle_name: str, angle_value: Optional[float]) -> None:
        """
        Add angle measurement to buffer
        
        Args:
            angle_name: Name/identifier of angle (e.g., 'knee_angle')
            angle_value: Measured angle in degrees
        """
        if angle_name not in self.angle_buffers:
            self.angle_buffers[angle_name] = deque(maxlen=self.window_size)
        
        if angle_value is not None:
            self.angle_buffers[angle_name].append(angle_value)
    
    def get_smoothed_angle(self, angle_name: str) -> Optional[float]:
        """
        Get smoothed angle value using moving average
        
        Args:
            angle_name: Name of angle to retrieve
        
        Returns:
            Smoothed angle value or None if insufficient data
        """
        if angle_name not in self.angle_buffers:
            return None
        
        buffer = self.angle_buffers[angle_name]
        
        if len(buffer) == 0:
            return None
        
        # Calculate moving average
        smoothed = np.mean(list(buffer))
        return float(smoothed)
    
    def add_landmark_measurement(
        self, 
        landmark_name: str, 
        x: float, 
        y: float, 
        z: float
    ) -> None:
        """
        Add landmark coordinate to buffer
        
        Args:
            landmark_name: Name of landmark (e.g., 'LEFT_KNEE')
            x, y, z: Coordinates
        """
        if landmark_name not in self.landmark_buffers:
            self.landmark_buffers[landmark_name] = deque(maxlen=self.window_size)
        
        self.landmark_buffers[landmark_name].append((x, y, z))
    
    def get_smoothed_landmark(self, landmark_name: str) -> Optional[tuple]:
        """
        Get smoothed landmark coordinates
        
        Returns:
            Tuple of (x, y, z) or None if insufficient data
        """
        if landmark_name not in self.landmark_buffers:
            return None
        
        buffer = self.landmark_buffers[landmark_name]
        
        if len(buffer) == 0:
            return None
        
        # Calculate average coordinates
        coords = np.array(list(buffer))
        smoothed = np.mean(coords, axis=0)
        
        return tuple(smoothed)
    
    def is_stable(self, angle_name: str, threshold: float = 5.0) -> bool:
        """
        Check if angle measurements are stable
        
        Args:
            angle_name: Name of angle to check
            threshold: Maximum standard deviation for stability (degrees)
        
        Returns:
            True if stable, False otherwise
        """
        if angle_name not in self.angle_buffers:
            return False
        
        buffer = self.angle_buffers[angle_name]
        
        if len(buffer) < self.window_size:
            return False  # Need full buffer for stability
        
        std_dev = np.std(list(buffer))
        return std_dev <= threshold
    
    def get_angle_variance(self, angle_name: str) -> Optional[float]:
        """
        Get variance of angle measurements
        Useful for detecting unstable movements
        
        Returns:
            Variance in degrees squared
        """
        if angle_name not in self.angle_buffers:
            return None
        
        buffer = self.angle_buffers[angle_name]
        
        if len(buffer) < 2:
            return None
        
        return float(np.var(list(buffer)))
    
    def reset(self) -> None:
        """Clear all buffers"""
        self.angle_buffers.clear()
        self.landmark_buffers.clear()
        logger.debug("Frame stabilizer buffers reset")
    
    def reset_angle(self, angle_name: str) -> None:
        """Reset specific angle buffer"""
        if angle_name in self.angle_buffers:
            self.angle_buffers[angle_name].clear()
    
    def reset_landmark(self, landmark_name: str) -> None:
        """Reset specific landmark buffer"""
        if landmark_name in self.landmark_buffers:
            self.landmark_buffers[landmark_name].clear()
    
    def is_buffer_full(self, angle_name: str) -> bool:
        """Check if angle buffer is full"""
        if angle_name not in self.angle_buffers:
            return False
        return len(self.angle_buffers[angle_name]) == self.window_size
    
    def get_buffer_size(self, angle_name: str) -> int:
        """Get current size of angle buffer"""
        if angle_name not in self.angle_buffers:
            return 0
        return len(self.angle_buffers[angle_name])
    
    def apply_median_filter(self, angle_name: str) -> Optional[float]:
        """
        Apply median filter instead of mean
        Better for rejecting outliers
        
        Returns:
            Median angle value
        """
        if angle_name not in self.angle_buffers:
            return None
        
        buffer = self.angle_buffers[angle_name]
        
        if len(buffer) == 0:
            return None
        
        median = np.median(list(buffer))
        return float(median)
    
    def detect_outlier(
        self, 
        angle_name: str, 
        new_value: float, 
        threshold: float = 20.0
    ) -> bool:
        """
        Detect if new value is an outlier
        
        Args:
            angle_name: Name of angle
            new_value: New measurement
            threshold: Maximum deviation from mean (degrees)
        
        Returns:
            True if outlier, False otherwise
        """
        if angle_name not in self.angle_buffers:
            return False  # No history, can't detect outlier
        
        buffer = self.angle_buffers[angle_name]
        
        if len(buffer) < 2:
            return False  # Need at least 2 samples
        
        mean = np.mean(list(buffer))
        deviation = abs(new_value - mean)
        
        return deviation > threshold
    
    def get_trend(self, angle_name: str) -> Optional[str]:
        """
        Get trend direction of angle
        
        Returns:
            'increasing', 'decreasing', or 'stable'
        """
        if angle_name not in self.angle_buffers:
            return None
        
        buffer = list(self.angle_buffers[angle_name])
        
        if len(buffer) < self.window_size:
            return None
        
        # Calculate linear regression slope
        x = np.arange(len(buffer))
        y = np.array(buffer)
        
        slope = np.polyfit(x, y, 1)[0]
        
        if abs(slope) < 1.0:  # Less than 1 degree/frame
            return 'stable'
        elif slope > 0:
            return 'increasing'
        else:
            return 'decreasing'
    
    def predict_next_value(self, angle_name: str) -> Optional[float]:
        """
        Predict next angle value using linear extrapolation
        Useful for anticipating movement
        
        Returns:
            Predicted angle value
        """
        if angle_name not in self.angle_buffers:
            return None
        
        buffer = list(self.angle_buffers[angle_name])
        
        if len(buffer) < 2:
            return None
        
        # Simple linear extrapolation
        x = np.arange(len(buffer))
        y = np.array(buffer)
        
        coeffs = np.polyfit(x, y, 1)
        next_x = len(buffer)
        predicted = coeffs[0] * next_x + coeffs[1]
        
        return float(predicted)


class AdaptiveStabilizer(FrameStabilizer):
    """
    Advanced stabilizer with adaptive window size
    Adjusts smoothing based on movement speed
    """
    
    def __init__(
        self, 
        min_window: int = 2, 
        max_window: int = 5,
        variance_threshold: float = 10.0
    ):
        """
        Initialize adaptive stabilizer
        
        Args:
            min_window: Minimum smoothing window
            max_window: Maximum smoothing window
            variance_threshold: Variance threshold for window adjustment
        """
        super().__init__(window_size=max_window)
        self.min_window = min_window
        self.max_window = max_window
        self.variance_threshold = variance_threshold
    
    def get_adaptive_smoothed_angle(self, angle_name: str) -> Optional[float]:
        """
        Get smoothed angle with adaptive window size
        Uses smaller window for fast movements, larger for slow
        """
        variance = self.get_angle_variance(angle_name)
        
        if variance is None:
            return self.get_smoothed_angle(angle_name)
        
        # Adjust window size based on variance
        if variance > self.variance_threshold:
            # High variance = fast movement = smaller window
            effective_window = self.min_window
        else:
            # Low variance = slow/stable = larger window
            effective_window = self.max_window
        
        if angle_name not in self.angle_buffers:
            return None
        
        buffer = list(self.angle_buffers[angle_name])
        
        if len(buffer) == 0:
            return None
        
        # Use only last N frames based on effective window
        recent_buffer = buffer[-effective_window:]
        smoothed = np.mean(recent_buffer)
        
        return float(smoothed)
