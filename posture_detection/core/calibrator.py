"""
User Body Calibrator
Measures user body proportions and adjusts exercise thresholds
"""

import numpy as np
from typing import Dict, Optional, Tuple
import logging
import time
from dataclasses import dataclass, asdict
import json

from utils.landmark_utils import LandmarkExtractor, LandmarkPoint
from core.angle_calculator import AngleCalculator

logger = logging.getLogger(__name__)


@dataclass
class BodyProportions:
    """User body proportion measurements"""
    torso_length: float  # Shoulder to hip distance
    upper_leg_length: float  # Hip to knee distance
    lower_leg_length: float  # Knee to ankle distance
    upper_arm_length: float  # Shoulder to elbow distance
    lower_arm_length: float  # Elbow to wrist distance
    shoulder_width: float  # Left to right shoulder distance
    hip_width: float  # Left to right hip distance
    total_height: float  # Estimated total height
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BodyProportions':
        """Create from dictionary"""
        return cls(**data)
    
    def save_to_file(self, filepath: str) -> None:
        """Save calibration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Calibration saved to {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'BodyProportions':
        """Load calibration from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        logger.info(f"Calibration loaded from {filepath}")
        return cls.from_dict(data)


@dataclass
class FlexibilityProfile:
    """User flexibility measurements"""
    squat_depth_angle: float  # Deepest comfortable squat knee angle
    hip_flexion_range: float  # Hip flexion range of motion
    shoulder_flexion_range: float  # Shoulder flexion range
    ankle_dorsiflexion: float  # Ankle dorsiflexion capacity
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FlexibilityProfile':
        """Create from dictionary"""
        return cls(**data)


class UserCalibrator:
    """
    Calibrate system to user's body proportions and flexibility
    """
    
    def __init__(self):
        """Initialize calibrator"""
        self.body_proportions: Optional[BodyProportions] = None
        self.flexibility_profile: Optional[FlexibilityProfile] = None
        self.is_calibrated = False
        
        self.calibration_samples = []
        self.calibration_duration = 3.0  # seconds
        
        logger.info("User calibrator initialized")
    
    def start_calibration(self) -> None:
        """Start calibration process"""
        self.calibration_samples = []
        self.is_calibrated = False
        logger.info("Calibration started")
    
    def add_calibration_frame(
        self,
        landmarks,
        image_width: int,
        image_height: int
    ) -> bool:
        """
        Add frame to calibration samples
        
        Args:
            landmarks: MediaPipe pose landmarks
            image_width: Frame width
            image_height: Frame height
        
        Returns:
            True if sample added successfully
        """
        try:
            # Extract key landmarks for body measurement
            extractor = LandmarkExtractor()
            
            required_landmarks = [
                'LEFT_SHOULDER', 'RIGHT_SHOULDER',
                'LEFT_HIP', 'RIGHT_HIP',
                'LEFT_KNEE', 'RIGHT_KNEE',
                'LEFT_ANKLE', 'RIGHT_ANKLE',
                'LEFT_ELBOW', 'RIGHT_ELBOW',
                'LEFT_WRIST', 'RIGHT_WRIST'
            ]
            
            landmark_points = extractor.extract_multiple_landmarks(
                landmarks, required_landmarks, image_width, image_height
            )
            
            # Check if all landmarks are visible
            if len(landmark_points) < len(required_landmarks):
                logger.warning("Not all landmarks visible for calibration")
                return False
            
            # Check confidence
            min_confidence = 0.8
            for point in landmark_points.values():
                if point.visibility < min_confidence:
                    logger.warning("Low landmark confidence for calibration")
                    return False
            
            self.calibration_samples.append(landmark_points)
            logger.debug(f"Calibration sample added (total: {len(self.calibration_samples)})")
            return True
            
        except Exception as e:
            logger.error(f"Error adding calibration frame: {e}")
            return False
    
    def complete_calibration(self) -> bool:
        """
        Complete calibration process
        Calculate average body proportions from samples
        
        Returns:
            True if calibration successful
        """
        if len(self.calibration_samples) < 5:
            logger.error(f"Insufficient calibration samples: {len(self.calibration_samples)}")
            return False
        
        try:
            # Calculate average measurements
            measurements = {
                'torso_length': [],
                'upper_leg_length': [],
                'lower_leg_length': [],
                'upper_arm_length': [],
                'lower_arm_length': [],
                'shoulder_width': [],
                'hip_width': [],
                'total_height': []
            }
            
            for sample in self.calibration_samples:
                # Torso length (shoulder to hip)
                left_shoulder = sample['LEFT_SHOULDER']
                left_hip = sample['LEFT_HIP']
                torso = LandmarkExtractor.calculate_distance(left_shoulder, left_hip)
                measurements['torso_length'].append(torso)
                
                # Upper leg (hip to knee)
                left_knee = sample['LEFT_KNEE']
                upper_leg = LandmarkExtractor.calculate_distance(left_hip, left_knee)
                measurements['upper_leg_length'].append(upper_leg)
                
                # Lower leg (knee to ankle)
                left_ankle = sample['LEFT_ANKLE']
                lower_leg = LandmarkExtractor.calculate_distance(left_knee, left_ankle)
                measurements['lower_leg_length'].append(lower_leg)
                
                # Upper arm (shoulder to elbow)
                left_elbow = sample['LEFT_ELBOW']
                upper_arm = LandmarkExtractor.calculate_distance(left_shoulder, left_elbow)
                measurements['upper_arm_length'].append(upper_arm)
                
                # Lower arm (elbow to wrist)
                left_wrist = sample['LEFT_WRIST']
                lower_arm = LandmarkExtractor.calculate_distance(left_elbow, left_wrist)
                measurements['lower_arm_length'].append(lower_arm)
                
                # Shoulder width
                right_shoulder = sample['RIGHT_SHOULDER']
                shoulder_width = LandmarkExtractor.calculate_distance(
                    left_shoulder, right_shoulder
                )
                measurements['shoulder_width'].append(shoulder_width)
                
                # Hip width
                right_hip = sample['RIGHT_HIP']
                hip_width = LandmarkExtractor.calculate_distance(left_hip, right_hip)
                measurements['hip_width'].append(hip_width)
                
                # Estimate total height
                total_height = torso + upper_leg + lower_leg
                measurements['total_height'].append(total_height)
            
            # Calculate averages
            avg_measurements = {
                key: np.mean(values) for key, values in measurements.items()
            }
            
            # Create body proportions object
            self.body_proportions = BodyProportions(**avg_measurements)
            
            self.is_calibrated = True
            logger.info("Calibration completed successfully")
            logger.info(f"Body proportions: {self.body_proportions}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error completing calibration: {e}")
            return False
    
    def calibrate_flexibility(
        self,
        exercise_name: str,
        landmarks,
        image_width: int,
        image_height: int
    ) -> bool:
        """
        Calibrate flexibility for specific exercise
        User performs exercise slowly to measure range of motion
        
        Args:
            exercise_name: Name of exercise for flexibility test
            landmarks: MediaPipe pose landmarks
            image_width: Frame width
            image_height: Frame height
        
        Returns:
            True if flexibility calibration successful
        """
        # This would be expanded to measure flexibility for each exercise
        # For now, using default flexibility values
        
        self.flexibility_profile = FlexibilityProfile(
            squat_depth_angle=80.0,  # Can squat to 80° knee angle
            hip_flexion_range=120.0,
            shoulder_flexion_range=170.0,
            ankle_dorsiflexion=25.0
        )
        
        logger.info("Flexibility profile set (using defaults)")
        return True
    
    def adjust_threshold_for_user(
        self,
        base_threshold: Tuple[float, float],
        adjustment_factor: float = 0.0
    ) -> Tuple[float, float]:
        """
        Adjust exercise threshold based on user calibration
        
        Args:
            base_threshold: (min, max) angle threshold
            adjustment_factor: Adjustment multiplier (0.0 = no adjustment)
        
        Returns:
            Adjusted (min, max) threshold
        """
        if not self.is_calibrated:
            return base_threshold
        
        min_angle, max_angle = base_threshold
        range_size = max_angle - min_angle
        
        # Adjust based on body proportions and flexibility
        adjustment = range_size * adjustment_factor
        
        return (min_angle - adjustment, max_angle + adjustment)
    
    def get_proportion_ratio(self, measurement_name: str) -> Optional[float]:
        """
        Get body proportion ratio normalized to height
        
        Args:
            measurement_name: Name of measurement (e.g., 'torso_length')
        
        Returns:
            Ratio value or None if not calibrated
        """
        if not self.is_calibrated or self.body_proportions is None:
            return None
        
        total_height = self.body_proportions.total_height
        if total_height == 0:
            return None
        
        measurement_value = getattr(self.body_proportions, measurement_name, None)
        if measurement_value is None:
            return None
        
        return measurement_value / total_height
    
    def is_body_type_tall(self, threshold: float = 1.2) -> bool:
        """Check if user has tall body type (relative proportions)"""
        if not self.is_calibrated:
            return False
        
        leg_ratio = self.get_proportion_ratio('upper_leg_length')
        if leg_ratio is None:
            return False
        
        return leg_ratio > threshold
    
    def is_body_type_flexible(self) -> bool:
        """Check if user has above-average flexibility"""
        if self.flexibility_profile is None:
            return False
        
        # Consider flexible if can achieve deeper squat
        return self.flexibility_profile.squat_depth_angle < 85.0
    
    def get_calibration_status(self) -> Dict[str, any]:
        """
        Get current calibration status
        
        Returns:
            Dictionary with calibration information
        """
        return {
            'is_calibrated': self.is_calibrated,
            'has_body_proportions': self.body_proportions is not None,
            'has_flexibility_profile': self.flexibility_profile is not None,
            'sample_count': len(self.calibration_samples),
            'body_proportions': self.body_proportions.to_dict() if self.body_proportions else None,
            'flexibility_profile': self.flexibility_profile.to_dict() if self.flexibility_profile else None
        }
    
    def reset(self) -> None:
        """Reset calibration"""
        self.body_proportions = None
        self.flexibility_profile = None
        self.is_calibrated = False
        self.calibration_samples = []
        logger.info("Calibration reset")


class CalibrationGuide:
    """Guide user through calibration process"""
    
    """Guide user through calibration process"""
    
    # Exercise-specific calibration poses
    EXERCISE_CALIBRATION = {
        'squat': [
            "Stand with feet shoulder-width apart",
            "Arms at your sides or crossed on chest",
            "Look straight ahead",
            "Hold steady for 5 seconds"
        ],
        'pushup': [
            "Get into high plank position",
            "Hands under shoulders, body straight",
            "Hold this starting position",
            "Keep body aligned for 5 seconds"
        ],
        'plank': [
            "Get into plank position (elbows down)",
            "Body in straight line",
            "Hold steady",
            "Maintain position for 5 seconds"
        ],
        'shoulder_press': [
            "Stand with feet shoulder-width apart",
            "Hold arms at shoulder level (like holding a barbell)",
            "Elbows at 90 degrees",
            "Hold this position for 5 seconds"
        ],
        'bicep_curl': [
            "Stand with feet shoulder-width apart",
            "Arms relaxed at your sides",
            "Look straight ahead",
            "Hold steady for 5 seconds"
        ],
        'general': [
            "Stand facing the camera in a neutral position",
            "Feet shoulder-width apart",
            "Arms relaxed at your sides",
            "Hold this position for 5 seconds"
        ]
    }
    
    @staticmethod
    def get_calibration_instructions(exercise_name: str = 'general') -> str:
        """Get calibration instruction text for specific exercise"""
        steps = CalibrationGuide.EXERCISE_CALIBRATION.get(
            exercise_name, 
            CalibrationGuide.EXERCISE_CALIBRATION['general']
        )
        return "\n".join([f"{i+1}. {step}" for i, step in enumerate(steps)])
    
    @staticmethod
    def get_current_step(elapsed_time: float, total_duration: float) -> str:
        """Get current calibration step message"""
        if elapsed_time < 1.0:
            return "Get into position..."
        elif elapsed_time < total_duration - 0.5:
            remaining = int(total_duration - elapsed_time)
            return f"Hold steady... {remaining}s remaining"
        else:
            return "Almost done!"
