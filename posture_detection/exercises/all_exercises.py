"""
All Exercise Implementations
Complete implementations of Squat, Pushup, Plank, Lunge, and Bicep Curl
"""

from typing import Dict, Optional
import logging
from exercises.base_exercise import BaseExercise, ExerciseMeasurements, FormFeedback
from core.angle_calculator import (
    AngleCalculator, calculate_knee_angle, calculate_hip_angle,
    calculate_elbow_angle, calculate_shoulder_angle
)

logger = logging.getLogger(__name__)


class SquatExercise(BaseExercise):
    """Squat exercise implementation"""
    
    def __init__(self):
        super().__init__('squat')
    
    def extract_measurements(self, landmarks, image_width: int, image_height: int) -> Optional[ExerciseMeasurements]:
        landmark_points = self.extract_landmarks(landmarks, image_width, image_height)
        if landmark_points is None:
            return None
        
        try:
            # Extract key points
            shoulder = landmark_points['LEFT_SHOULDER']
            hip = landmark_points['LEFT_HIP']
            knee = landmark_points['LEFT_KNEE']
            ankle = landmark_points['LEFT_ANKLE']
            
            # Calculate angles
            knee_angle = calculate_knee_angle(hip, knee, ankle)
            hip_angle = calculate_hip_angle(shoulder, hip, knee)
            back_angle = AngleCalculator.calculate_angle_from_vertical(shoulder, hip)
            shin_angle = AngleCalculator.calculate_shin_angle(knee, ankle)
            
            if None in [knee_angle, hip_angle, back_angle]:
                return None
            
            angles = {
                'knee_angle': knee_angle,
                'hip_angle': hip_angle,
                'back_angle': back_angle,
                'shin_angle': shin_angle if shin_angle else 0
            }
            
            phase = self.determine_phase(ExerciseMeasurements(
                angles=angles, alignments={}, distances={}, phase='', is_valid=True
            ))
            
            return ExerciseMeasurements(
                angles=angles,
                alignments={},
                distances={},
                phase=phase,
                is_valid=True
            )
        except Exception as e:
            logger.error(f"Error extracting squat measurements: {e}")
            return None
    
    def determine_phase(self, measurements: ExerciseMeasurements) -> str:
        knee_angle = measurements.angles.get('knee_angle', 180)
        
        if knee_angle >= 160:
            return 'standing'
        elif knee_angle >= 110:
            return 'descent'
        elif knee_angle >= 80:
            return 'bottom'
        else:
            return 'ascent'
    
    def validate_form(self, measurements: ExerciseMeasurements) -> FormFeedback:
        messages = []
        corrections = []
        
        knee_angle = measurements.angles.get('knee_angle')
        hip_angle = measurements.angles.get('hip_angle')
        back_angle = measurements.angles.get('back_angle')
        
        phase = measurements.phase
        
        if phase == 'bottom':
            # Check squat depth
            base_range = self.get_angle_range('bottom', 'knee_angle')
            adjusted_range = self.adjust_threshold_for_user(
                base_range, 
                getattr(self, 'calibrator', None)
            )
            if knee_angle and not (adjusted_range[0] <= knee_angle <= adjusted_range[1]):
                if knee_angle > 115:
                    corrections.append("Squat deeper - knees should reach 80-110°")
                elif knee_angle < 75:
                    corrections.append("Don't go too deep - maintain control")
            
            # Check hip angle
            if hip_angle and not self.is_angle_in_range(hip_angle, 'bottom', 'hip_angle'):
                corrections.append("Adjust hip position - maintain proper depth")
            
            # Check back angle
            if back_angle and back_angle > 50:
                corrections.append("Keep chest up - reduce forward lean")
        
        is_correct = len(corrections) == 0
        severity = 'good' if is_correct else ('warning' if len(corrections) == 1 else 'error')
        
        if is_correct:
            messages.append("Excellent squat form!")
        
        return FormFeedback(
            is_correct=is_correct,
            messages=messages,
            corrections=corrections,
            severity=severity
        )


class PushupExercise(BaseExercise):
    """Push-up exercise implementation"""
    
    def __init__(self):
        super().__init__('pushup')
    
    def extract_measurements(self, landmarks, image_width: int, image_height: int) -> Optional[ExerciseMeasurements]:
        landmark_points = self.extract_landmarks(landmarks, image_width, image_height)
        if landmark_points is None:
            return None
        
        try:
            shoulder = landmark_points['LEFT_SHOULDER']
            elbow = landmark_points['LEFT_ELBOW']
            wrist = landmark_points['LEFT_WRIST']
            hip = landmark_points['LEFT_HIP']
            ankle = landmark_points['LEFT_ANKLE']
            
            elbow_angle = calculate_elbow_angle(shoulder, elbow, wrist)
            body_alignment = AngleCalculator.calculate_body_alignment(shoulder, hip, ankle)
            
            if None in [elbow_angle, body_alignment]:
                return None
            
            angles = {
                'elbow_angle': elbow_angle,
                'body_alignment': body_alignment
            }
            
            phase = self.determine_phase(ExerciseMeasurements(
                angles=angles, alignments={}, distances={}, phase='', is_valid=True
            ))
            
            return ExerciseMeasurements(
                angles=angles,
                alignments={'body': body_alignment},
                distances={},
                phase=phase,
                is_valid=True
            )
        except Exception as e:
            logger.error(f"Error extracting pushup measurements: {e}")
            return None
    
    def determine_phase(self, measurements: ExerciseMeasurements) -> str:
        elbow_angle = measurements.angles.get('elbow_angle', 180)
        
        if elbow_angle >= 155:
            return 'up'
        elif elbow_angle >= 90:
            return 'descent'
        elif elbow_angle >= 65:
            return 'down'
        else:
            return 'ascent'
    
    def validate_form(self, measurements: ExerciseMeasurements) -> FormFeedback:
        messages = []
        corrections = []
        
        elbow_angle = measurements.angles.get('elbow_angle')
        body_alignment = measurements.alignments.get('body')
        
        phase = measurements.phase
        
        if phase == 'down':
            if elbow_angle and not self.is_angle_in_range(elbow_angle, 'down', 'elbow_angle'):
                if elbow_angle > 95:
                    corrections.append("Go lower - elbows should reach 70-90°")
                elif elbow_angle < 65:
                    corrections.append("Don't go too low - maintain form")
        
        if body_alignment and body_alignment > 15:
            if body_alignment > 20:
                corrections.append("Keep body straight - no sagging or piking")
            else:
                corrections.append("Minor body alignment issue")
        
        is_correct = len(corrections) == 0
        severity = 'good' if is_correct else 'warning'
        
        if is_correct:
            messages.append("Perfect push-up form!")
        
        return FormFeedback(
            is_correct=is_correct,
            messages=messages,
            corrections=corrections,
            severity=severity
        )


class PlankExercise(BaseExercise):
    """Plank exercise implementation"""
    
    def __init__(self):
        super().__init__('plank')
    
    def extract_measurements(self, landmarks, image_width: int, image_height: int) -> Optional[ExerciseMeasurements]:
        landmark_points = self.extract_landmarks(landmarks, image_width, image_height)
        if landmark_points is None:
            return None
        
        try:
            shoulder = landmark_points['LEFT_SHOULDER']
            hip = landmark_points['LEFT_HIP']
            ankle = landmark_points['LEFT_ANKLE']
            elbow = landmark_points['LEFT_ELBOW']
            
            body_alignment = AngleCalculator.calculate_body_alignment(shoulder, hip, ankle)
            hip_angle = calculate_hip_angle(shoulder, hip, ankle)
            
            if None in [body_alignment, hip_angle]:
                return None
            
            angles = {
                'hip_angle': hip_angle,
                'body_alignment': body_alignment
            }
            
            return ExerciseMeasurements(
                angles=angles,
                alignments={'body': body_alignment},
                distances={},
                phase='hold',
                is_valid=True
            )
        except Exception as e:
            logger.error(f"Error extracting plank measurements: {e}")
            return None
    
    def determine_phase(self, measurements: ExerciseMeasurements) -> str:
        return 'hold'
    
    def validate_form(self, measurements: ExerciseMeasurements) -> FormFeedback:
        messages = []
        corrections = []
        
        hip_angle = measurements.angles.get('hip_angle')
        body_alignment = measurements.alignments.get('body')
        
        if hip_angle:
            if hip_angle < 155:
                corrections.append("Hips sagging - engage core and lift")
            elif hip_angle > 185:
                corrections.append("Hips too high - lower to straight line")
        
        if body_alignment and body_alignment > 15:
            corrections.append("Maintain straight line from head to heels")
        
        is_correct = len(corrections) == 0
        severity = 'good' if is_correct else 'error'
        
        if is_correct:
            messages.append("Great plank hold!")
        
        return FormFeedback(
            is_correct=is_correct,
            messages=messages,
            corrections=corrections,
            severity=severity
        )
    
    
class ShoulderPressExercise(BaseExercise):
    """Shoulder press (overhead press) exercise implementation"""
    
    def __init__(self):
        super().__init__('shoulder_press')
    
    def extract_measurements(self, landmarks, image_width: int, image_height: int) -> Optional[ExerciseMeasurements]:
        landmark_points = self.extract_landmarks(landmarks, image_width, image_height)
        if landmark_points is None:
            return None
        
        try:
            shoulder = landmark_points['LEFT_SHOULDER']
            elbow = landmark_points['LEFT_ELBOW']
            wrist = landmark_points['LEFT_WRIST']
            hip = landmark_points['LEFT_HIP']
            
            # Calculate angles
            elbow_angle = calculate_elbow_angle(shoulder, elbow, wrist)
            shoulder_flexion = calculate_shoulder_angle(hip, shoulder, elbow)
            elbow_to_body = AngleCalculator.calculate_elbow_to_body_angle(shoulder, elbow, hip)
            
            if elbow_angle is None:
                return None
            
            angles = {
                'elbow_angle': elbow_angle,
                'shoulder_flexion': shoulder_flexion if shoulder_flexion else 0,
                'elbow_to_body_angle': elbow_to_body if elbow_to_body else 30
            }
            
            phase = self.determine_phase(ExerciseMeasurements(
                angles=angles, alignments={}, distances={}, phase='', is_valid=True
            ))
            
            return ExerciseMeasurements(
                angles=angles,
                alignments={},
                distances={},
                phase=phase,
                is_valid=True
            )
        except Exception as e:
            logger.error(f"Error extracting shoulder press measurements: {e}")
            return None
    
    def determine_phase(self, measurements: ExerciseMeasurements) -> str:
        elbow_angle = measurements.angles.get('elbow_angle', 90)
        
        if elbow_angle <= 105:
            return 'starting'
        elif elbow_angle <= 165:
            return 'pressing'
        elif elbow_angle >= 165:
            return 'lockout'
        else:
            return 'lowering'
    
    def validate_form(self, measurements: ExerciseMeasurements) -> FormFeedback:
        messages = []
        corrections = []
        
        elbow_angle = measurements.angles.get('elbow_angle')
        elbow_to_body = measurements.angles.get('elbow_to_body_angle')
        phase = measurements.phase
        
        if phase == 'lockout':
            if elbow_angle and elbow_angle < 165:
                corrections.append("Fully extend arms overhead - reach lockout")
        elif phase == 'starting':
            if elbow_angle and not self.is_angle_in_range(elbow_angle, 'starting', 'elbow_angle'):
                corrections.append("Start with elbows at shoulder level (90°)")
        
        if elbow_to_body and (elbow_to_body < 20 or elbow_to_body > 55):
            corrections.append("Keep elbows at 30-45° angle - don't flare wide")
        
        is_correct = len(corrections) == 0
        severity = 'good' if is_correct else 'warning'
        
        if is_correct:
            messages.append("Excellent shoulder press form!")
        
        return FormFeedback(
            is_correct=is_correct,
            messages=messages,
            corrections=corrections,
            severity=severity
        )




class BicepCurlExercise(BaseExercise):
    """Bicep curl exercise implementation"""
    
    def __init__(self):
        super().__init__('bicep_curl')
    
    def extract_measurements(self, landmarks, image_width: int, image_height: int) -> Optional[ExerciseMeasurements]:
        landmark_points = self.extract_landmarks(landmarks, image_width, image_height)
        if landmark_points is None:
            return None
        
        try:
            shoulder = landmark_points['LEFT_SHOULDER']
            elbow = landmark_points['LEFT_ELBOW']
            wrist = landmark_points['LEFT_WRIST']
            
            elbow_angle = calculate_elbow_angle(shoulder, elbow, wrist)
            
            if elbow_angle is None:
                return None
            
            angles = {'elbow_angle': elbow_angle}
            
            phase = self.determine_phase(ExerciseMeasurements(
                angles=angles, alignments={}, distances={}, phase='', is_valid=True
            ))
            
            return ExerciseMeasurements(
                angles=angles,
                alignments={},
                distances={},
                phase=phase,
                is_valid=True
            )
        except Exception as e:
            logger.error(f"Error extracting bicep curl measurements: {e}")
            return None
    
    def determine_phase(self, measurements: ExerciseMeasurements) -> str:
        elbow_angle = measurements.angles.get('elbow_angle', 180)
        
        if elbow_angle >= 165:
            return 'extended'
        elif elbow_angle >= 90:
            return 'lifting'
        elif elbow_angle >= 25:
            return 'contracted'
        else:
            return 'lowering'
    
    def validate_form(self, measurements: ExerciseMeasurements) -> FormFeedback:
        messages = []
        corrections = []
        
        elbow_angle = measurements.angles.get('elbow_angle')
        phase = measurements.phase
        
        if phase == 'contracted':
            if elbow_angle and elbow_angle > 50:
                corrections.append("Curl higher - full contraction")
        elif phase == 'extended':
            if elbow_angle and elbow_angle < 165:
                corrections.append("Fully extend arms at bottom")
        
        is_correct = len(corrections) == 0
        severity = 'good' if is_correct else 'warning'
        
        if is_correct:
            messages.append("Good curl form!")
        
        return FormFeedback(
            is_correct=is_correct,
            messages=messages,
            corrections=corrections,
            severity=severity
        )
