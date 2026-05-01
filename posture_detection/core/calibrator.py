"""
User Body Calibrator
Measures user body proportions from a single good frame.
No sample collection - just extract measurements when landmarks are clean.
"""

import numpy as np
from typing import Dict, Optional, Tuple
import logging
import json
from dataclasses import dataclass, asdict

from utils.landmark_utils import LandmarkExtractor, LandmarkPoint
from core.angle_calculator import AngleCalculator

logger = logging.getLogger(__name__)


@dataclass
class BodyProportions:
    """User body proportion measurements (all in pixels, relative to frame)"""
    torso_length: float
    upper_leg_length: float
    lower_leg_length: float
    upper_arm_length: float
    lower_arm_length: float
    shoulder_width: float
    hip_width: float
    total_height: float

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'BodyProportions':
        return cls(**data)

    def save_to_file(self, filepath: str) -> None:
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Calibration saved to {filepath}")

    @classmethod
    def load_from_file(cls, filepath: str) -> 'BodyProportions':
        with open(filepath, 'r') as f:
            data = json.load(f)
        logger.info(f"Calibration loaded from {filepath}")
        return cls.from_dict(data)


@dataclass
class FlexibilityProfile:
    """User flexibility - uses safe defaults, no measurement needed"""
    squat_depth_angle: float = 80.0
    hip_flexion_range: float = 120.0
    shoulder_flexion_range: float = 170.0
    ankle_dorsiflexion: float = 25.0

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'FlexibilityProfile':
        return cls(**data)


class UserCalibrator:
    """
    Simplified calibrator - measures body proportions from ONE good frame.
    No sample collection, no retries, no minimum counts.
    """

    # Landmarks needed for body measurement
    REQUIRED_LANDMARKS = [
        'LEFT_SHOULDER', 'RIGHT_SHOULDER',
        'LEFT_HIP', 'RIGHT_HIP',
        'LEFT_KNEE', 'RIGHT_KNEE',
        'LEFT_ANKLE', 'RIGHT_ANKLE',
        'LEFT_ELBOW', 'RIGHT_ELBOW',
        'LEFT_WRIST', 'RIGHT_WRIST'
    ]

    # Minimum visibility for a landmark to be considered valid
    MIN_VISIBILITY = 0.5  # Lowered from 0.7 - more lenient

    def __init__(self):
        self.body_proportions: Optional[BodyProportions] = None
        self.flexibility_profile: Optional[FlexibilityProfile] = None
        self.is_calibrated = False
        logger.info("UserCalibrator initialized (single-frame mode)")

    def reset(self) -> None:
        """Reset calibration state"""
        self.body_proportions = None
        self.flexibility_profile = None
        self.is_calibrated = False
        logger.info("Calibration reset")

    def check_frame_quality(
        self,
        landmarks,
        image_width: int,
        image_height: int
    ) -> Tuple[bool, float, str]:
        """
        Check if frame has good landmarks AND person is standing upright.
        Rejects sitting, crouching, lying down poses.
        """
        try:
            extractor = LandmarkExtractor()
            points = extractor.extract_multiple_landmarks(
                landmarks, self.REQUIRED_LANDMARKS, image_width, image_height
            )

            visible_count = len(points)
            total = len(self.REQUIRED_LANDMARKS)
            confident_count = sum(
                1 for p in points.values()
                if p.visibility >= self.MIN_VISIBILITY
            )
            avg_vis = (
                sum(p.visibility for p in points.values()) / len(points)
                if points else 0.0
            )
            quality_score = (confident_count / total) * 100.0

            # --- Check 1: Enough landmarks visible ---
            if confident_count < 8:
                missing = total - confident_count
                return (False, quality_score,
                        f"Step back - {missing} body parts not visible. Full body must be in frame.")

            # --- Check 2: Person must be STANDING (vertical body check) ---
            # Get key landmarks for pose validation
            l_shoulder = points.get('LEFT_SHOULDER')
            r_shoulder = points.get('RIGHT_SHOULDER')
            l_hip = points.get('LEFT_HIP')
            r_hip = points.get('RIGHT_HIP')
            l_knee = points.get('LEFT_KNEE')
            r_knee = points.get('RIGHT_KNEE')
            l_ankle = points.get('LEFT_ANKLE')
            r_ankle = points.get('RIGHT_ANKLE')

            # Use whichever side is more visible
            shoulder = l_shoulder if (l_shoulder and l_shoulder.visibility > 0.5) else r_shoulder
            hip = l_hip if (l_hip and l_hip.visibility > 0.5) else r_hip
            knee = l_knee if (l_knee and l_knee.visibility > 0.5) else r_knee
            ankle = l_ankle if (l_ankle and l_ankle.visibility > 0.5) else r_ankle

            if not all([shoulder, hip, knee, ankle]):
                return (False, quality_score,
                        "Cannot detect full body - ensure head to feet are visible.")

            # In image coordinates: Y increases DOWNWARD
            # So for standing: shoulder.y < hip.y < knee.y < ankle.y

            # Check vertical ordering (shoulders above hips above knees above ankles)
            if not (shoulder.y < hip.y):
                return (False, quality_score,
                        "Please STAND UP straight - shoulders must be above hips.")

            if not (hip.y < knee.y):
                return (False, quality_score,
                        "Please STAND UP - you appear to be sitting or crouching too low.")

            if not (knee.y < ankle.y):
                return (False, quality_score,
                        "Please STAND UP - full legs must be visible and upright.")

            # Check legs are reasonably straight (not bent/sitting)
            # Knee angle: hip->knee->ankle. When standing straight this is ~170-180°
            from core.angle_calculator import calculate_knee_angle
            knee_angle = calculate_knee_angle(hip, knee, ankle)
            if knee_angle is not None and knee_angle < 140:
                return (False, quality_score,
                        f"Please STAND STRAIGHT - legs appear bent ({knee_angle:.0f}°). Straighten up.")

            # Check body is roughly vertical (not lying sideways)
            # Shoulder to hip vector should be mostly vertical
            body_dx = abs(shoulder.x - hip.x)
            body_dy = abs(shoulder.y - hip.y)
            if body_dy < body_dx:
                return (False, quality_score,
                        "Please face the camera standing upright - body appears tilted sideways.")

            # Check person fills reasonable portion of frame (not too far)
            body_height_ratio = abs(ankle.y - shoulder.y) / image_height
            if body_height_ratio < 0.35:
                return (False, quality_score,
                        "Step closer to camera - you are too far away.")

            if body_height_ratio > 0.97:
                return (False, quality_score,
                        "Step back from camera - too close, full body not visible.")

            return (True, quality_score,
                    f"Good standing position! Hold still... ({confident_count}/{total} landmarks)")

        except Exception as e:
            logger.error(f"Error checking frame quality: {e}")
            return (False, 0.0, "Detection error - ensure full body is visible in good lighting.")
    def calibrate_from_frame(
        self,
        landmarks,
        image_width: int,
        image_height: int
    ) -> bool:
        """
        Extract body measurements from a single frame.
        Call this once you have a good quality frame.

        Returns:
            True if calibration succeeded
        """
        try:
            extractor = LandmarkExtractor()
            points = extractor.extract_multiple_landmarks(
                landmarks, self.REQUIRED_LANDMARKS, image_width, image_height
            )

            if len(points) < 8:
                logger.error(f"Not enough landmarks: {len(points)}/{len(self.REQUIRED_LANDMARKS)}")
                return False

            # Helper - safe distance (returns 0 if either point missing)
            def dist(a, b):
                pa = points.get(a)
                pb = points.get(b)
                if pa and pb:
                    return LandmarkExtractor.calculate_distance(pa, pb)
                return 0.0

            torso = dist('LEFT_SHOULDER', 'LEFT_HIP')
            upper_leg = dist('LEFT_HIP', 'LEFT_KNEE')
            lower_leg = dist('LEFT_KNEE', 'LEFT_ANKLE')
            upper_arm = dist('LEFT_SHOULDER', 'LEFT_ELBOW')
            lower_arm = dist('LEFT_ELBOW', 'LEFT_WRIST')
            shoulder_width = dist('LEFT_SHOULDER', 'RIGHT_SHOULDER')
            hip_width = dist('LEFT_HIP', 'RIGHT_HIP')

            # If left side missing, try right side
            if torso == 0:
                torso = dist('RIGHT_SHOULDER', 'RIGHT_HIP')
            if upper_leg == 0:
                upper_leg = dist('RIGHT_HIP', 'RIGHT_KNEE')
            if lower_leg == 0:
                lower_leg = dist('RIGHT_KNEE', 'RIGHT_ANKLE')
            if upper_arm == 0:
                upper_arm = dist('RIGHT_SHOULDER', 'RIGHT_ELBOW')
            if lower_arm == 0:
                lower_arm = dist('RIGHT_ELBOW', 'RIGHT_WRIST')

            total_height = torso + upper_leg + lower_leg
            if total_height < 10:
                logger.error("Body measurements too small - person may be too far from camera")
                return False

            self.body_proportions = BodyProportions(
                torso_length=torso,
                upper_leg_length=upper_leg,
                lower_leg_length=lower_leg,
                upper_arm_length=upper_arm,
                lower_arm_length=lower_arm,
                shoulder_width=shoulder_width,
                hip_width=hip_width,
                total_height=total_height
            )

            # Always use safe default flexibility profile
            self.flexibility_profile = FlexibilityProfile()

            self.is_calibrated = True

            logger.info(
                f"Calibration complete - height={total_height:.0f}px, "
                f"torso={torso:.0f}px, upper_leg={upper_leg:.0f}px, "
                f"shoulder_width={shoulder_width:.0f}px"
            )
            return True

        except Exception as e:
            logger.error(f"Error during calibration: {e}")
            return False

    def get_proportion_ratio(self, measurement_name: str) -> Optional[float]:
        """Get body proportion ratio normalized to total height"""
        if not self.is_calibrated or self.body_proportions is None:
            return None
        total_height = self.body_proportions.total_height
        if total_height == 0:
            return None
        value = getattr(self.body_proportions, measurement_name, None)
        if value is None:
            return None
        return value / total_height

    def is_body_type_flexible(self) -> bool:
        """Always False with default profile - no flexibility test done"""
        return False

    def is_body_type_tall(self, threshold: float = 1.2) -> bool:
        """Check if user has longer-than-average legs"""
        if not self.is_calibrated:
            return False
        leg_ratio = self.get_proportion_ratio('upper_leg_length')
        if leg_ratio is None:
            return False
        return leg_ratio > threshold

    def get_calibration_status(self) -> Dict:
        return {
            'is_calibrated': self.is_calibrated,
            'has_body_proportions': self.body_proportions is not None,
            'has_flexibility_profile': self.flexibility_profile is not None,
            'body_proportions': self.body_proportions.to_dict() if self.body_proportions else None,
        }


class CalibrationGuide:
    """Instructions shown to user during calibration"""

    EXERCISE_CALIBRATION = {
        'squat': [
            "Stand with feet shoulder-width apart",
            "Arms relaxed at your sides",
            "Look straight ahead at the camera",
            "Make sure full body is visible"
        ],
        'pushup': [
            "Stand upright facing the camera",
            "Arms relaxed at your sides",
            "Step back so full body is in frame",
            "Hold still for 3 seconds"
        ],
        'plank': [
            "Stand upright facing the camera",
            "Arms relaxed at your sides",
            "Full body must be in frame",
            "Hold still for 3 seconds"
        ],
        'shoulder_press': [
            "Stand with feet shoulder-width apart",
            "Arms relaxed at your sides",
            "Look straight ahead",
            "Make sure full body is visible"
        ],
        'bicep_curl': [
            "Stand with feet shoulder-width apart",
            "Arms relaxed at your sides",
            "Look straight ahead at the camera",
            "Hold still for 3 seconds"
        ],
        'general': [
            "Stand facing the camera",
            "Feet shoulder-width apart",
            "Arms relaxed at your sides",
            "Make sure full body is visible"
        ]
    }

    @staticmethod
    def get_calibration_instructions(exercise_name: str = 'general') -> str:
        steps = CalibrationGuide.EXERCISE_CALIBRATION.get(
            exercise_name,
            CalibrationGuide.EXERCISE_CALIBRATION['general']
        )
        return "\n".join([f"{i+1}. {step}" for i, step in enumerate(steps)])

    @staticmethod
    def get_current_step(elapsed_time: float, total_duration: float) -> str:
        if elapsed_time < 1.0:
            return "Get into position..."
        elif elapsed_time < total_duration - 0.5:
            remaining = int(total_duration - elapsed_time)
            return f"Hold still... {remaining}s remaining"
        else:
            return "Almost done!"
