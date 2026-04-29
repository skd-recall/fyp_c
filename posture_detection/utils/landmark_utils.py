"""
Landmark Utilities for MediaPipe Pose
Handles landmark extraction and coordinate transformations
"""

from typing import Optional, Tuple, List
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# MediaPipe Pose Landmark Indices
mp_pose = mp.solutions.pose


@dataclass
class LandmarkPoint:
    """Represents a single landmark point with coordinates and confidence"""
    x: float
    y: float
    z: float
    visibility: float
    
    def to_tuple(self) -> Tuple[float, float]:
        """Convert to (x, y) tuple for 2D operations"""
        return (self.x, self.y)
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array"""
        return np.array([self.x, self.y, self.z])


class LandmarkExtractor:
    """Extract and process MediaPipe landmarks"""
    
    # Landmark name to index mapping
    LANDMARK_MAP = {
        'NOSE': mp_pose.PoseLandmark.NOSE,
        'LEFT_EYE_INNER': mp_pose.PoseLandmark.LEFT_EYE_INNER,
        'LEFT_EYE': mp_pose.PoseLandmark.LEFT_EYE,
        'LEFT_EYE_OUTER': mp_pose.PoseLandmark.LEFT_EYE_OUTER,
        'RIGHT_EYE_INNER': mp_pose.PoseLandmark.RIGHT_EYE_INNER,
        'RIGHT_EYE': mp_pose.PoseLandmark.RIGHT_EYE,
        'RIGHT_EYE_OUTER': mp_pose.PoseLandmark.RIGHT_EYE_OUTER,
        'LEFT_EAR': mp_pose.PoseLandmark.LEFT_EAR,
        'RIGHT_EAR': mp_pose.PoseLandmark.RIGHT_EAR,
        'MOUTH_LEFT': mp_pose.PoseLandmark.MOUTH_LEFT,
        'MOUTH_RIGHT': mp_pose.PoseLandmark.MOUTH_RIGHT,
        'LEFT_SHOULDER': mp_pose.PoseLandmark.LEFT_SHOULDER,
        'RIGHT_SHOULDER': mp_pose.PoseLandmark.RIGHT_SHOULDER,
        'LEFT_ELBOW': mp_pose.PoseLandmark.LEFT_ELBOW,
        'RIGHT_ELBOW': mp_pose.PoseLandmark.RIGHT_ELBOW,
        'LEFT_WRIST': mp_pose.PoseLandmark.LEFT_WRIST,
        'RIGHT_WRIST': mp_pose.PoseLandmark.RIGHT_WRIST,
        'LEFT_PINKY': mp_pose.PoseLandmark.LEFT_PINKY,
        'RIGHT_PINKY': mp_pose.PoseLandmark.RIGHT_PINKY,
        'LEFT_INDEX': mp_pose.PoseLandmark.LEFT_INDEX,
        'RIGHT_INDEX': mp_pose.PoseLandmark.RIGHT_INDEX,
        'LEFT_THUMB': mp_pose.PoseLandmark.LEFT_THUMB,
        'RIGHT_THUMB': mp_pose.PoseLandmark.RIGHT_THUMB,
        'LEFT_HIP': mp_pose.PoseLandmark.LEFT_HIP,
        'RIGHT_HIP': mp_pose.PoseLandmark.RIGHT_HIP,
        'LEFT_KNEE': mp_pose.PoseLandmark.LEFT_KNEE,
        'RIGHT_KNEE': mp_pose.PoseLandmark.RIGHT_KNEE,
        'LEFT_ANKLE': mp_pose.PoseLandmark.LEFT_ANKLE,
        'RIGHT_ANKLE': mp_pose.PoseLandmark.RIGHT_ANKLE,
        'LEFT_HEEL': mp_pose.PoseLandmark.LEFT_HEEL,
        'RIGHT_HEEL': mp_pose.PoseLandmark.RIGHT_HEEL,
        'LEFT_FOOT_INDEX': mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
        'RIGHT_FOOT_INDEX': mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
    }
    
    @staticmethod
    def get_landmark_index(landmark_name: str) -> int:
        """Get MediaPipe index for landmark name"""
        if landmark_name not in LandmarkExtractor.LANDMARK_MAP:
            raise ValueError(f"Unknown landmark: {landmark_name}")
        return LandmarkExtractor.LANDMARK_MAP[landmark_name].value
    
    @staticmethod
    def extract_landmark(
        landmarks,
        landmark_name: str,
        image_width: int,
        image_height: int
    ) -> Optional[LandmarkPoint]:
        """
        Extract a single landmark point
        
        Args:
            landmarks: MediaPipe pose landmarks
            landmark_name: Name of landmark (e.g., 'LEFT_SHOULDER')
            image_width: Width of image in pixels
            image_height: Height of image in pixels
        
        Returns:
            LandmarkPoint or None if not found/confident
        """
        try:
            idx = LandmarkExtractor.get_landmark_index(landmark_name)
            landmark = landmarks.landmark[idx]
            
            return LandmarkPoint(
                x=landmark.x * image_width,
                y=landmark.y * image_height,
                z=landmark.z,
                visibility=landmark.visibility
            )
        except (IndexError, AttributeError) as e:
            logger.warning(f"Failed to extract {landmark_name}: {e}")
            return None
    
    @staticmethod
    def extract_multiple_landmarks(
        landmarks,
        landmark_names: List[str],
        image_width: int,
        image_height: int
    ) -> dict:
        """
        Extract multiple landmarks
        
        Returns:
            Dictionary mapping landmark names to LandmarkPoint objects
        """
        result = {}
        for name in landmark_names:
            point = LandmarkExtractor.extract_landmark(
                landmarks, name, image_width, image_height
            )
            if point is not None:
                result[name] = point
        return result
    
    @staticmethod
    def calculate_distance(point1: LandmarkPoint, point2: LandmarkPoint) -> float:
        """Calculate Euclidean distance between two landmarks"""
        return np.sqrt(
            (point2.x - point1.x) ** 2 + 
            (point2.y - point1.y) ** 2
        )
    
    @staticmethod
    def calculate_midpoint(
        point1: LandmarkPoint, 
        point2: LandmarkPoint
    ) -> LandmarkPoint:
        """Calculate midpoint between two landmarks"""
        return LandmarkPoint(
            x=(point1.x + point2.x) / 2,
            y=(point1.y + point2.y) / 2,
            z=(point1.z + point2.z) / 2,
            visibility=min(point1.visibility, point2.visibility)
        )
    
    @staticmethod
    def is_landmark_visible(
        landmark: Optional[LandmarkPoint],
        confidence_threshold: float = 0.7
    ) -> bool:
        """Check if landmark is visible and confident"""
        if landmark is None:
            return False
        return landmark.visibility >= confidence_threshold
    
    @staticmethod
    def normalize_coordinates(
        point: LandmarkPoint,
        reference_distance: float
    ) -> LandmarkPoint:
        """
        Normalize landmark coordinates by reference distance
        Useful for body proportion calibration
        """
        return LandmarkPoint(
            x=point.x / reference_distance,
            y=point.y / reference_distance,
            z=point.z / reference_distance,
            visibility=point.visibility
        )
    
    @staticmethod
    def get_body_center(landmarks, image_width: int, image_height: int) -> Optional[LandmarkPoint]:
        """Calculate approximate body center (hip midpoint)"""
        left_hip = LandmarkExtractor.extract_landmark(
            landmarks, 'LEFT_HIP', image_width, image_height
        )
        right_hip = LandmarkExtractor.extract_landmark(
            landmarks, 'RIGHT_HIP', image_width, image_height
        )
        
        if left_hip and right_hip:
            return LandmarkExtractor.calculate_midpoint(left_hip, right_hip)
        return None


def get_landmark_connections() -> List[Tuple[str, str]]:
    """Get landmark connections for visualization"""
    return [
        # Torso
        ('LEFT_SHOULDER', 'RIGHT_SHOULDER'),
        ('LEFT_SHOULDER', 'LEFT_HIP'),
        ('RIGHT_SHOULDER', 'RIGHT_HIP'),
        ('LEFT_HIP', 'RIGHT_HIP'),
        
        # Left arm
        ('LEFT_SHOULDER', 'LEFT_ELBOW'),
        ('LEFT_ELBOW', 'LEFT_WRIST'),
        
        # Right arm
        ('RIGHT_SHOULDER', 'RIGHT_ELBOW'),
        ('RIGHT_ELBOW', 'RIGHT_WRIST'),
        
        # Left leg
        ('LEFT_HIP', 'LEFT_KNEE'),
        ('LEFT_KNEE', 'LEFT_ANKLE'),
        
        # Right leg
        ('RIGHT_HIP', 'RIGHT_KNEE'),
        ('RIGHT_KNEE', 'RIGHT_ANKLE'),
    ]
class CoordinateNormalizer:
    """
    Normalize landmark coordinates relative to body center
    Allows detection of off-center users
    """
    
    @staticmethod
    def normalize_to_body_center(
        landmarks_dict: dict,
        image_width: int,
        image_height: int
    ) -> dict:
        """
        Normalize all landmarks relative to body center
        
        Args:
            landmarks_dict: Dictionary of landmark name -> LandmarkPoint
            image_width: Image width
            image_height: Image height
        
        Returns:
            Normalized landmarks dictionary
        """
        # Find body center (hip midpoint)
        if 'LEFT_HIP' in landmarks_dict and 'RIGHT_HIP' in landmarks_dict:
            center_x = (landmarks_dict['LEFT_HIP'].x + landmarks_dict['RIGHT_HIP'].x) / 2
            center_y = (landmarks_dict['LEFT_HIP'].y + landmarks_dict['RIGHT_HIP'].y) / 2
        elif 'LEFT_SHOULDER' in landmarks_dict and 'RIGHT_SHOULDER' in landmarks_dict:
            # Fallback to shoulder center
            center_x = (landmarks_dict['LEFT_SHOULDER'].x + landmarks_dict['RIGHT_SHOULDER'].x) / 2
            center_y = (landmarks_dict['LEFT_SHOULDER'].y + landmarks_dict['RIGHT_SHOULDER'].y) / 2
        else:
            # Can't normalize - return as-is
            return landmarks_dict
        
        # Calculate body size for scaling
        body_height = CoordinateNormalizer._estimate_body_height(landmarks_dict)
        
        if body_height == 0:
            return landmarks_dict
        
        # Normalize each landmark
        normalized = {}
        for name, point in landmarks_dict.items():
            # Translate to center
            norm_x = (point.x - center_x) / body_height
            norm_y = (point.y - center_y) / body_height
            
            # Keep original visibility and z
            normalized[name] = LandmarkPoint(
                x=norm_x * image_width / 2 + image_width / 2,  # Re-center to image
                y=norm_y * image_height / 2 + image_height / 2,
                z=point.z,
                visibility=point.visibility
            )
        
        return normalized
    
    @staticmethod
    def _estimate_body_height(landmarks_dict: dict) -> float:
        """Estimate body height from landmarks"""
        # Try shoulder to ankle
        if 'LEFT_SHOULDER' in landmarks_dict and 'LEFT_ANKLE' in landmarks_dict:
            return abs(landmarks_dict['LEFT_SHOULDER'].y - landmarks_dict['LEFT_ANKLE'].y)
        
        # Try hip to ankle
        if 'LEFT_HIP' in landmarks_dict and 'LEFT_ANKLE' in landmarks_dict:
            return abs(landmarks_dict['LEFT_HIP'].y - landmarks_dict['LEFT_ANKLE'].y) * 1.5
        
        return 0.0
