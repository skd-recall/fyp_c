"""
Base Exercise Class
Abstract base class defining interface for all exercise implementations
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import logging

from utils.landmark_utils import LandmarkExtractor, LandmarkPoint
from core.angle_calculator import AngleCalculator
from config.config_manager import config_manager

logger = logging.getLogger(__name__)


@dataclass
class ExerciseMeasurements:
    """Container for exercise-specific measurements"""
    angles: Dict[str, float]  # Measured angles
    alignments: Dict[str, float]  # Body alignments
    distances: Dict[str, float]  # Distances between landmarks
    phase: str  # Current exercise phase
    is_valid: bool  # Whether measurements are valid


@dataclass
class FormFeedback:
    """Feedback for exercise form"""
    is_correct: bool
    messages: List[str]  # List of feedback messages
    corrections: List[str]  # List of specific corrections needed
    severity: str  # 'good', 'warning', 'error'


class BaseExercise(ABC):
    """
    Abstract base class for all exercises
    Defines common interface and shared functionality
    """
    
    def __init__(self, exercise_name: str):
        """
        Initialize exercise
        
        Args:
            exercise_name: Name of exercise (matches config key)
        """
        self.exercise_name = exercise_name
        self.config = config_manager.get_exercise_config(exercise_name)
        self.key_landmarks = config_manager.get_key_landmarks(exercise_name)
        self.phases = config_manager.get_exercise_phases(exercise_name)
        self.form_checks = config_manager.get_form_checks(exercise_name)
        
        self.current_phase = None
        self.measurements_history = []
        
        logger.info(f"Exercise '{exercise_name}' initialized")
    
    @abstractmethod
    def extract_measurements(
        self,
        landmarks,
        image_width: int,
        image_height: int
    ) -> Optional[ExerciseMeasurements]:
        """
        Extract exercise-specific measurements from landmarks
        Must be implemented by each exercise subclass
        
        Args:
            landmarks: MediaPipe pose landmarks
            image_width: Frame width
            image_height: Frame height
        
        Returns:
            ExerciseMeasurements object or None if extraction fails
        """
        pass
    
    @abstractmethod
    def determine_phase(
        self,
        measurements: ExerciseMeasurements
    ) -> str:
        """
        Determine current exercise phase from measurements
        
        Args:
            measurements: Current exercise measurements
        
        Returns:
            Phase name (e.g., 'standing', 'descent', 'bottom', 'ascent')
        """
        pass
    
    @abstractmethod
    def validate_form(
        self,
        measurements: ExerciseMeasurements
    ) -> FormFeedback:
        """
        Validate exercise form against rules
        
        Args:
            measurements: Current exercise measurements
        
        Returns:
            FormFeedback object with corrections
        """
        pass
    
    def extract_landmarks(
        self,
        landmarks,
        image_width: int,
        image_height: int
    ) -> Optional[Dict[str, LandmarkPoint]]:
        """
        Extract key landmarks for this exercise
        
        Returns:
            Dictionary of landmark name to LandmarkPoint
        """
        extractor = LandmarkExtractor()
        landmark_points = extractor.extract_multiple_landmarks(
            landmarks, self.key_landmarks, image_width, image_height
        )
        
        # Check if all key landmarks are present
        if len(landmark_points) < len(self.key_landmarks):
            missing = set(self.key_landmarks) - set(landmark_points.keys())
            logger.warning(f"Missing landmarks for {self.exercise_name}: {missing}")
            return None
        
        # Check confidence threshold
        confidence_threshold = config_manager.get_confidence_threshold()
        for name, point in landmark_points.items():
            if point.visibility < confidence_threshold:
                logger.warning(
                    f"Low confidence for {name}: {point.visibility:.2f}"
                )
                return None
        
        return landmark_points
    
    def is_angle_in_range(
        self,
        angle: float,
        phase: str,
        angle_key: str,
        tolerance: float = 5.0
    ) -> bool:
        """
        Check if angle is within acceptable range for phase
        
        Args:
            angle: Measured angle
            phase: Current exercise phase
            angle_key: Key for angle range in config
            tolerance: Additional tolerance in degrees
        
        Returns:
            True if angle is valid
        """
        return config_manager.validate_angle_range(
            angle, angle_key, self.exercise_name, phase, tolerance
        )
    def adjust_threshold_for_user(
        self,
        base_range: tuple,
        calibrator = None
    ) -> tuple:
        """
        Adjust angle threshold based on user calibration
    
        Args:
            base_range: (min, max) base threshold
            calibrator: UserCalibrator instance with body data
    
        Returns:
            Adjusted (min, max) threshold
        """
        if calibrator is None or not calibrator.is_calibrated:
            return base_range  # No adjustment
    
        min_angle, max_angle = base_range
    
    # Example: If user has longer limbs, allow slightly wider range
    # This is a simple adjustment - can be made more sophisticated
        flexibility_factor = 1.0
    
        if calibrator.flexibility_profile:
        # If user is flexible, allow deeper ranges
            if calibrator.is_body_type_flexible():
                flexibility_factor = 1.1  # 10% wider range
    
    # Apply adjustment
        range_size = max_angle - min_angle
        adjustment = (range_size * (flexibility_factor - 1.0)) / 2
    
        return (min_angle - adjustment, max_angle + adjustment)
    
    def get_angle_range(
        self,
        phase: str,
        angle_key: str,
        tolerance: float = 5.0
    ) -> Tuple[float, float]:
        """Get angle range with tolerance for phase"""
        return config_manager.get_angle_range_with_tolerance(
            self.exercise_name, phase, angle_key, tolerance
        )
    
    def add_measurement(self, measurement: ExerciseMeasurements) -> None:
        """Add measurement to history"""
        self.measurements_history.append(measurement)
        # Keep only last 30 measurements
        if len(self.measurements_history) > 30:
            self.measurements_history.pop(0)
    
    def get_recent_measurements(self, count: int = 5) -> List[ExerciseMeasurements]:
        """Get most recent measurements"""
        return self.measurements_history[-count:]
    
    def reset_history(self) -> None:
        """Clear measurement history"""
        self.measurements_history = []
        self.current_phase = None
        logger.debug(f"History reset for {self.exercise_name}")
    
    def get_exercise_info(self) -> Dict:
        """Get exercise information"""
        return {
            'name': self.exercise_name,
            'display_name': self.config.get('name', self.exercise_name),
            'camera_view': self.config.get('camera_view', 'side'),
            'phases': list(self.phases.keys()),
            'key_landmarks': self.key_landmarks
        }
    
    def get_rep_counting_config(self) -> Dict:
        """Get rep counting configuration"""
        return config_manager.get_rep_counting_config(self.exercise_name)
    
    def format_feedback_message(
        self,
        messages: List[str],
        max_messages: int = 3
    ) -> str:
        """
        Format feedback messages for display
        
        Args:
            messages: List of feedback messages
            max_messages: Maximum messages to show
        
        Returns:
            Formatted message string
        """
        if not messages:
            return "Form looks good!"
        
        # Limit number of messages
        messages = messages[:max_messages]
        
        if len(messages) == 1:
            return messages[0]
        
        # Format as numbered list
        return "\n".join([f"{i+1}. {msg}" for i, msg in enumerate(messages)])
    
    def calculate_form_score(
        self,
        measurements: ExerciseMeasurements
    ) -> float:
        """
        Calculate overall form score (0-100)
        
        Args:
            measurements: Current measurements
        
        Returns:
            Score from 0-100
        """
        feedback = self.validate_form(measurements)
        
        if feedback.is_correct:
            return 100.0
        
        # Deduct points for each issue
        total_issues = len(feedback.corrections)
        if total_issues == 0:
            return 100.0
        
        # Severity-based deductions
        severity_deductions = {
            'error': 20,
            'warning': 10,
            'good': 0
        }
        
        deduction = severity_deductions.get(feedback.severity, 10)
        score = max(0, 100 - (deduction * total_issues))
        
        return score
    
    def is_ready_to_start(
        self,
        landmarks,
        image_width: int,
        image_height: int
    ) -> bool:
        """
        Check if user is in correct starting position
        
        Returns:
            True if ready to start exercise
        """
        measurements = self.extract_measurements(landmarks, image_width, image_height)
        
        if measurements is None:
            return False
        
        phase = self.determine_phase(measurements)
        
        # Check if in starting phase (typically 'standing' or 'up' or 'extended')
        starting_phases = ['standing', 'up', 'extended', 'hold']
        return phase in starting_phases


def validate_landmarks_visible(
    landmarks_dict: Dict[str, LandmarkPoint],
    required_landmarks: List[str],
    confidence_threshold: float = 0.7
) -> Tuple[bool, List[str]]:
    """
    Validate that all required landmarks are visible and confident
    
    Returns:
        Tuple of (all_visible, list_of_missing_landmarks)
    """
    missing = []
    
    for landmark_name in required_landmarks:
        if landmark_name not in landmarks_dict:
            missing.append(landmark_name)
            continue
        
        point = landmarks_dict[landmark_name]
        if point.visibility < confidence_threshold:
            missing.append(f"{landmark_name} (low confidence)")
    
    return len(missing) == 0, missing
