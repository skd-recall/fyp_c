"""
Posture Detection System - Main Application
Real-time exercise form detection and feedback
"""

import cv2
import time
import logging
from typing import Optional

from core.pose_detector import PoseDetector, CameraManager
from core.frame_stabilizer import FrameStabilizer
from core.calibrator import UserCalibrator, CalibrationGuide
from exercises.all_exercises import (
    SquatExercise, PushupExercise, PlankExercise,
    ShoulderPressExercise, BicepCurlExercise
)
from validation.rep_counter import RepCounter
from feedback.feedback_engine import FeedbackEngine
from config.config_manager import config_manager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PostureDetectionSystem:
    """Main posture detection system"""
    
    EXERCISE_MAP = {
        'squat': SquatExercise,
        'pushup': PushupExercise,
        'plank': PlankExercise,
        'shoulder_press': ShoulderPressExercise,
        'bicep_curl': BicepCurlExercise
    }
    
    def __init__(self, camera_id: int = 1):
        """Initialize system"""
        logger.info("Initializing Posture Detection System")
        
        # Core components
        self.pose_detector = PoseDetector(
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.camera_manager = CameraManager(camera_id, width=640, height=480)
        self.frame_stabilizer = FrameStabilizer(
            window_size=config_manager.get_smoothing_frames()
        )
        self.calibrator = UserCalibrator()
        self.feedback_engine = FeedbackEngine(
            cooldown_seconds=config_manager.get_feedback_cooldown()
        )
        
        # State
        self.current_exercise = None
        self.rep_counter = None
        self.is_running = False
        self.is_calibrated = False
        
        logger.info("System initialized successfully")
    
    def start_calibration(self) -> bool:
        """Start user calibration process"""
        logger.info("Starting calibration")
        
        if not self.camera_manager.open():
            logger.error("Failed to open camera")
            return False
        exercise_name = self.current_exercise.exercise_name if self.current_exercise else "general"
        
        self.calibrator.start_calibration()
        calibration_duration = 5.0
        start_time = time.time()
        frame_count = 0
        
        while True:
            ret, frame = self.camera_manager.read_frame()
            if not ret:
                continue
            
            # Detect pose
            result = self.pose_detector.detect_pose(frame)
            
            if result and result.success:
                # Draw landmarks
                frame = self.pose_detector.draw_landmarks(frame, result.landmarks)
                
                # Add calibration frame
                elapsed = time.time() - start_time
                if elapsed < calibration_duration:
                    self.calibrator.add_calibration_frame(
                        result.landmarks, result.frame_width, result.frame_height
                    )
                    frame_count += 1
                    
                    # Show exercise-specific instructions
                    if elapsed < 1.0:
                    # Show full instructions at start
                        exercise_name = self.current_exercise.exercise_name if self.current_exercise else "general"
                        instructions = CalibrationGuide.get_calibration_instructions(exercise_name)
                        y_pos = 30
                        for line in instructions.split('\n'):
                            self.pose_detector.draw_text(
                                frame, line, (10, y_pos), font_scale=0.5
                            )
                            y_pos += 25
                    else:
                    # Show countdown
                        instruction = CalibrationGuide.get_current_step(
                            elapsed, calibration_duration
                        )
                        self.pose_detector.draw_text(
                            frame, instruction, (10, 30), font_scale=0.7, 
                            color=(0, 255, 0)
                        )
                else:
                    # Complete calibration
                    if self.calibrator.complete_calibration():
                        self.is_calibrated = True
                        logger.info("Calibration completed successfully")
                        time.sleep(1)
                        break
                    else:
                        logger.error("Calibration failed")
                        return False
            
            cv2.imshow("Calibration", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Calibration cancelled")
                break
        
        cv2.destroyAllWindows()
        self.camera_manager.release()
        return self.is_calibrated
    
    def select_exercise(self, exercise_name: str) -> bool:
        """Select exercise to perform"""
        exercise_name = exercise_name.lower().replace(" ", "_")
    
        if exercise_name not in self.EXERCISE_MAP:
            logger.error(f"Unknown exercise: {exercise_name}")
            return False
    
        self.current_exercise = self.EXERCISE_MAP[exercise_name]()
    
        # Give exercise access to calibrator
        self.current_exercise.calibrator = self.calibrator
        rep_config = self.current_exercise.get_rep_counting_config()
        self.rep_counter = RepCounter(exercise_name, rep_config)
        
        logger.info(f"Exercise selected: {exercise_name}")
        return True
    
    def run_exercise_session(self):
        """Run exercise detection session"""
        if self.current_exercise is None:
            logger.error("No exercise selected")
            return
        
        if not self.camera_manager.open():
            logger.error("Failed to open camera")
            return
        
        self.is_running = True
        self.frame_stabilizer.reset()
        self.rep_counter.reset()
        
        start_time = time.time()
        frame_count = 0
        
        logger.info(f"Starting exercise session: {self.current_exercise.exercise_name}")
        
        while self.is_running:
            ret, frame = self.camera_manager.read_frame()
            if not ret:
                continue
            
            frame_count += 1
            
            # Detect pose
            result = self.pose_detector.detect_pose(frame)
            
            if result and result.success:
                # Extract measurements
                measurements = self.current_exercise.extract_measurements(
                    result.landmarks, result.frame_width, result.frame_height
                )
                
                if measurements:
                    # Smooth angles
                    for angle_name, angle_value in measurements.angles.items():
                        self.frame_stabilizer.add_angle_measurement(angle_name, angle_value)
                    
                    # Get primary angle for rep counting
                    primary_angle_key = self._get_primary_angle_key()
                    smoothed_angle = self.frame_stabilizer.get_smoothed_angle(primary_angle_key)
                    
                    if smoothed_angle is not None:
                        # Update rep counter
                        rep_status = self.rep_counter.update(smoothed_angle)
                        
                        # Validate form
                        form_feedback = self.current_exercise.validate_form(measurements)
                        form_score = self.current_exercise.calculate_form_score(measurements)
                        
                        # Generate feedback
                        feedback_message = self.feedback_engine.generate_feedback(
                            form_feedback.corrections,
                            rep_status['rep_count'],
                            form_score
                        )
                        
                        # Draw on frame
                        frame = self._draw_feedback(
                            frame, measurements, rep_status, form_score, feedback_message
                        )
                    
                # Draw landmarks
                frame = self.pose_detector.draw_landmarks(frame, result.landmarks)
            
            # Calculate and display FPS
            current_time = time.time()
            fps = self.pose_detector.calculate_fps(frame_count, start_time, current_time)
            frame = self.pose_detector.draw_fps(frame, fps)
            
            # Show frame
            cv2.imshow("Posture Detection", frame)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.is_running = False
            elif key == ord('r'):
                self.rep_counter.reset()
                logger.info("Rep counter reset")
        
        # Cleanup
        cv2.destroyAllWindows()
        self.camera_manager.release()
        logger.info("Exercise session ended")
    
    def _get_primary_angle_key(self) -> str:
        """Get primary angle for rep counting"""
        exercise_type = self.current_exercise.exercise_name
        angle_map = {
            'squat': 'knee_angle',
            'pushup': 'elbow_angle',
            'plank': 'body_alignment',
            'shoulder_press': 'elbow_angle',
            'bicep_curl': 'elbow_angle'
        }
        return angle_map.get(exercise_type, 'knee_angle')
    
    def _draw_feedback(
        self, frame, measurements, rep_status, form_score, feedback_message
    ):
        """Draw feedback on frame"""
        y_offset = 30
        line_height = 30
        
        # Rep count
        self.pose_detector.draw_text(
            frame, f"Reps: {rep_status['rep_count']}", (10, y_offset)
        )
        y_offset += line_height
        
        # Phase
        self.pose_detector.draw_text(
            frame, f"Phase: {measurements.phase}", (10, y_offset)
        )
        y_offset += line_height
        
        # Form score
        color = (0, 255, 0) if form_score > 80 else (0, 255, 255) if form_score > 60 else (0, 0, 255)
        self.pose_detector.draw_text(
            frame, f"Form: {form_score:.0f}%", (10, y_offset), color=color
        )
        y_offset += line_height
        
        # Feedback message
        if feedback_message:
            self.pose_detector.draw_text(
                frame, feedback_message, (10, y_offset), font_scale=0.5
            )
        
        return frame


def main():
    """Main entry point"""
    print("=== AI-Based Gym Trainer: Posture Detection System ===\n")
    
    # Initialize system
    system = PostureDetectionSystem(camera_id=0)
    
    # Select exercise FIRST
    print("Available exercises:")
    print("1. Squat")
    print("2. Push-up")
    print("3. Plank")
    print("4. Shoulder Press")
    print("5. Bicep Curl")
    
    choice = input("\nSelect exercise (1-5): ")
    exercise_map = {
        '1': 'squat',
        '2': 'pushup',
        '3': 'plank',
        '4': 'shoulder_press',
        '5': 'bicep_curl'
    }
    
    exercise_name = exercise_map.get(choice, 'squat')
    if not system.select_exercise(exercise_name):
        print("Failed to select exercise. Exiting.")
        return
    
    print(f"\nExercise selected: {exercise_name}")
    
    # NOW do calibration (exercise-specific)
    print("\n=== Step 1: Calibration ===")
    print(f"We'll calibrate your body for {exercise_name}")
    input("Press Enter to start calibration...")
    
    if not system.start_calibration():
        print("Calibration failed. Exiting.")
        return
    
    print("\nCalibration successful!")
    
    exercise_name = exercise_map.get(choice, 'squat')
    if not system.select_exercise(exercise_name):
        print("Failed to select exercise. Exiting.")
        return
    
    print(f"\nExercise selected: {exercise_name}")
    print("\nInstructions:")
    print("- Stand in camera view")
    print("- Press 'Q' to quit")
    print("- Press 'R' to reset rep counter")
    input("\nPress Enter to start...")
    
    # Run session
    system.run_exercise_session()
    
    print("\nSession completed. Thank you!")


if __name__ == "__main__":
    main()
