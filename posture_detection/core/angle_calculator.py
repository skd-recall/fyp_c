"""
Angle Calculator for Biomechanical Analysis
Calculates joint angles, body alignment, and other geometric measurements
"""

import numpy as np
from typing import Optional, Tuple
import logging
from utils.landmark_utils import LandmarkPoint

logger = logging.getLogger(__name__)


class AngleCalculator:
    """Calculate angles and alignments from landmarks"""
    
    @staticmethod
    def calculate_angle(
        point1: LandmarkPoint,
        vertex: LandmarkPoint,
        point3: LandmarkPoint
    ) -> Optional[float]:
        """
        Calculate angle at vertex formed by three points
        
        Args:
            point1: First point
            vertex: Vertex point (angle is calculated here)
            point3: Third point
        
        Returns:
            Angle in degrees (0-180) or None if calculation fails
        
        Formula:
            angle = arccos((v1 · v2) / (|v1| * |v2|))
            where v1 = point1 - vertex, v2 = point3 - vertex
        """
        try:
            # Create vectors
            v1 = np.array([point1.x - vertex.x, point1.y - vertex.y])
            v2 = np.array([point3.x - vertex.x, point3.y - vertex.y])
            
            # Calculate angle using dot product
            dot_product = np.dot(v1, v2)
            magnitude_v1 = np.linalg.norm(v1)
            magnitude_v2 = np.linalg.norm(v2)
            
            # Avoid division by zero
            if magnitude_v1 == 0 or magnitude_v2 == 0:
                logger.warning("Zero magnitude vector in angle calculation")
                return None
            
            # Calculate cosine of angle
            cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
            
            # Clamp to [-1, 1] to avoid numerical errors
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            
            # Calculate angle in radians then convert to degrees
            angle_rad = np.arccos(cos_angle)
            angle_deg = np.degrees(angle_rad)
            
            return float(angle_deg)
            
        except Exception as e:
            logger.error(f"Error calculating angle: {e}")
            return None
    
    @staticmethod
    def calculate_angle_from_vertical(
        point1: LandmarkPoint,
        point2: LandmarkPoint
    ) -> Optional[float]:
        """
        Calculate angle of line segment from vertical (y-axis)
        Useful for measuring back lean, body tilt, etc.
        
        Args:
            point1: Starting point (e.g., shoulder)
            point2: Ending point (e.g., hip)
        
        Returns:
            Angle from vertical in degrees (0-90)
        """
        try:
            # Calculate vector
            dx = point2.x - point1.x
            dy = point2.y - point1.y
            
            # Calculate angle from vertical (y-axis)
            # atan2 gives angle from horizontal, so subtract from 90
            angle_from_horizontal = np.degrees(np.arctan2(dy, dx))
            angle_from_vertical = abs(90 - abs(angle_from_horizontal))
            
            return float(angle_from_vertical)
            
        except Exception as e:
            logger.error(f"Error calculating angle from vertical: {e}")
            return None
    
    @staticmethod
    def calculate_angle_from_horizontal(
        point1: LandmarkPoint,
        point2: LandmarkPoint
    ) -> Optional[float]:
        """
        Calculate angle of line segment from horizontal (x-axis)
        
        Returns:
            Angle from horizontal in degrees
        """
        try:
            dx = point2.x - point1.x
            dy = point2.y - point1.y
            
            angle = np.degrees(np.arctan2(dy, dx))
            return float(angle)
            
        except Exception as e:
            logger.error(f"Error calculating angle from horizontal: {e}")
            return None
    
    @staticmethod
    def calculate_body_alignment(
        shoulder: LandmarkPoint,
        hip: LandmarkPoint,
        ankle: LandmarkPoint
    ) -> Optional[float]:
        """
        Calculate body alignment deviation from straight line
        Returns deviation in degrees (0 = perfect alignment)
        
        Used for push-ups, planks, etc.
        """
        try:
            # Calculate angle at hip
            angle = AngleCalculator.calculate_angle(shoulder, hip, ankle)
            
            if angle is None:
                return None
            
            # Perfect alignment is 180°, return deviation
            deviation = abs(180 - angle)
            return float(deviation)
            
        except Exception as e:
            logger.error(f"Error calculating body alignment: {e}")
            return None
    
    @staticmethod
    def calculate_knee_valgus(
        hip: LandmarkPoint,
        knee: LandmarkPoint,
        ankle: LandmarkPoint
    ) -> Optional[float]:
        """
        Calculate knee valgus (inward collapse)
        Returns horizontal deviation of knee from hip-ankle line
        
        Positive value = knee caving inward
        Negative value = knee bowing outward
        """
        try:
            # Calculate perpendicular distance from knee to hip-ankle line
            # Using point-to-line distance formula
            
            # Line from hip to ankle
            x1, y1 = hip.x, hip.y
            x2, y2 = ankle.x, ankle.y
            x0, y0 = knee.x, knee.y
            
            # Calculate perpendicular distance
            numerator = abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1))
            denominator = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            if denominator == 0:
                return None
            
            distance = numerator / denominator
            
            # Determine direction (inward = positive, outward = negative)
            # Cross product to determine side
            cross = (x2 - x1) * (y0 - y1) - (y2 - y1) * (x0 - x1)
            sign = 1 if cross > 0 else -1
            
            return float(distance * sign)
            
        except Exception as e:
            logger.error(f"Error calculating knee valgus: {e}")
            return None
    
    @staticmethod
    def calculate_shin_angle(
        knee: LandmarkPoint,
        ankle: LandmarkPoint
    ) -> Optional[float]:
        """
        Calculate shin angle from vertical
        Used for squat and lunge form checking
        
        Returns:
            Angle in degrees (90 = horizontal, 0 = vertical)
        """
        return AngleCalculator.calculate_angle_from_vertical(knee, ankle)
    
    @staticmethod
    def calculate_elbow_to_body_angle(
        shoulder: LandmarkPoint,
        elbow: LandmarkPoint,
        hip: LandmarkPoint
    ) -> Optional[float]:
        """
        Calculate angle between elbow and torso
        Used for push-up form (should be 20-45°)
        
        Returns:
            Angle in degrees
        """
        try:
            # Vector from shoulder to elbow
            v1 = np.array([elbow.x - shoulder.x, elbow.y - shoulder.y])
            
            # Vector from shoulder to hip (torso)
            v2 = np.array([hip.x - shoulder.x, hip.y - shoulder.y])
            
            # Calculate angle
            dot_product = np.dot(v1, v2)
            mag1 = np.linalg.norm(v1)
            mag2 = np.linalg.norm(v2)
            
            if mag1 == 0 or mag2 == 0:
                return None
            
            cos_angle = np.clip(dot_product / (mag1 * mag2), -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            
            return float(angle)
            
        except Exception as e:
            logger.error(f"Error calculating elbow-to-body angle: {e}")
            return None
    
    @staticmethod
    def calculate_shoulder_flexion(
        shoulder: LandmarkPoint,
        hip: LandmarkPoint,
        elbow: LandmarkPoint
    ) -> Optional[float]:
        """
        Calculate shoulder flexion angle
        Used for bicep curl, overhead press, etc.
        
        Returns:
            Angle in degrees (0 = arm at side, 90 = arm horizontal, 180 = arm overhead)
        """
        try:
            # Reference vertical line from shoulder downward
            vertical_point = LandmarkPoint(
                x=shoulder.x,
                y=shoulder.y + 100,  # Point below shoulder
                z=shoulder.z,
                visibility=1.0
            )
            
            angle = AngleCalculator.calculate_angle(
                vertical_point, shoulder, elbow
            )
            
            return angle
            
        except Exception as e:
            logger.error(f"Error calculating shoulder flexion: {e}")
            return None
    
    @staticmethod
    def normalize_angle_range(angle: float) -> float:
        """
        Normalize angle to 0-180 range
        
        Args:
            angle: Angle in degrees
        
        Returns:
            Normalized angle
        """
        if angle is None:
            return None
        
        # Ensure angle is in 0-360 range
        angle = angle % 360
        
        # Convert to 0-180 range
        if angle > 180:
            angle = 360 - angle
        
        return angle
    
    @staticmethod
    def calculate_angle_velocity(
        current_angle: float,
        previous_angle: float,
        time_delta: float
    ) -> float:
        """
        Calculate angular velocity (degrees per second)
        Useful for detecting movement speed
        
        Args:
            current_angle: Current angle measurement
            previous_angle: Previous angle measurement
            time_delta: Time difference in seconds
        
        Returns:
            Angular velocity in degrees/second
        """
        if time_delta == 0:
            return 0.0
        
        angle_change = current_angle - previous_angle
        velocity = angle_change / time_delta
        
        return velocity


# Convenience functions for common calculations

def calculate_knee_angle(hip: LandmarkPoint, knee: LandmarkPoint, ankle: LandmarkPoint) -> Optional[float]:
    """Calculate knee flexion/extension angle"""
    return AngleCalculator.calculate_angle(hip, knee, ankle)


def calculate_hip_angle(shoulder: LandmarkPoint, hip: LandmarkPoint, knee: LandmarkPoint) -> Optional[float]:
    """Calculate hip flexion/extension angle"""
    return AngleCalculator.calculate_angle(shoulder, hip, knee)


def calculate_elbow_angle(shoulder: LandmarkPoint, elbow: LandmarkPoint, wrist: LandmarkPoint) -> Optional[float]:
    """Calculate elbow flexion/extension angle"""
    return AngleCalculator.calculate_angle(shoulder, elbow, wrist)


def calculate_shoulder_angle(hip: LandmarkPoint, shoulder: LandmarkPoint, elbow: LandmarkPoint) -> Optional[float]:
    """Calculate shoulder angle"""
    return AngleCalculator.calculate_angle(hip, shoulder, elbow)


def calculate_ankle_angle(knee: LandmarkPoint, ankle: LandmarkPoint, foot: LandmarkPoint) -> Optional[float]:
    """Calculate ankle dorsiflexion/plantarflexion"""
    return AngleCalculator.calculate_angle(knee, ankle, foot)
