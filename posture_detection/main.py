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
    
    def __init__(self, camera_id: int = 0):
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
        """
        Simplified calibration - show a 5-second countdown, then grab one
        clean frame to extract body proportions. No sample collection.
        """
        logger.info("Starting calibration")

        if not self.camera_manager.open():
            logger.error("Failed to open camera")
            return False

        exercise_name = self.current_exercise.exercise_name if self.current_exercise else "general"
        instructions = CalibrationGuide.get_calibration_instructions(exercise_name)
        instruction_lines = instructions.split('\n')

        calibration_duration = 5.0  # seconds to hold pose
        start_time = time.time()
        calibration_done = False

        while True:
            ret, frame = self.camera_manager.read_frame()
            if not ret:
                continue

            elapsed = time.time() - start_time
            remaining = max(0, calibration_duration - elapsed)

            result = self.pose_detector.detect_pose(frame)

            if result and result.success:
                frame = self.pose_detector.draw_landmarks(frame, result.landmarks)

                # Check quality every frame and show feedback
                is_good, quality, message = self.calibrator.check_frame_quality(
                    result.landmarks, result.frame_width, result.frame_height
                )

                # Color feedback: green = good, red = bad
                msg_color = (0, 255, 0) if is_good else (0, 0, 255)
                self.pose_detector.draw_text(
                    frame, message, (10, 30), font_scale=0.6,
                    color=msg_color, background=True, background_color=(0, 0, 0)
                )

                # Quality bar
                bar_width = int(quality * 3)
                bar_color = (0, 255, 0) if quality > 70 else (0, 255, 255) if quality > 50 else (0, 0, 255)
                cv2.rectangle(frame, (10, 50), (10 + bar_width, 68), bar_color, -1)
                cv2.rectangle(frame, (10, 50), (310, 68), (255, 255, 255), 1)

                # Countdown timer
                timer_text = f"Hold still: {remaining:.1f}s"
                self.pose_detector.draw_text(
                    frame, timer_text, (10, 85), font_scale=0.6,
                    color=(255, 255, 255), background=True, background_color=(0, 0, 0)
                )

                # If time is up and quality is good - calibrate now
                if elapsed >= calibration_duration and is_good:
                    success = self.calibrator.calibrate_from_frame(
                        result.landmarks, result.frame_width, result.frame_height
                    )
                    if success:
                        self.is_calibrated = True
                        self.pose_detector.draw_text(
                            frame, "Calibration Complete!", (10, 110),
                            font_scale=0.8, color=(0, 255, 0),
                            background=True, background_color=(0, 0, 0)
                        )
                        cv2.imshow("Calibration", frame)
                        cv2.waitKey(1000)
                        calibration_done = True
                        break
                    else:
                        # Bad frame at end - reset timer and try again
                        start_time = time.time()
                        logger.warning("Frame not good enough at end - resetting timer")

                elif elapsed >= calibration_duration and not is_good:
                    # Time up but bad quality - reset timer
                    start_time = time.time()
                    self.pose_detector.draw_text(
                        frame, "Restarting - step back so full body is visible",
                        (10, 110), font_scale=0.5, color=(0, 100, 255),
                        background=True, background_color=(0, 0, 0)
                    )

            else:
                # No pose detected
                self.pose_detector.draw_text(
                    frame, "No person detected - step into frame",
                    (10, 30), font_scale=0.7, color=(0, 0, 255),
                    background=True, background_color=(0, 0, 0)
                )
                # Reset timer when no person visible
                start_time = time.time()

            # Show instructions on the right side
            for i, line in enumerate(instruction_lines):
                self.pose_detector.draw_text(
                    frame, line, (frame.shape[1] - 320, 30 + i * 22),
                    font_scale=0.45, color=(200, 200, 200),
                    background=False
                )

            cv2.imshow("Calibration", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Calibration cancelled by user")
                cv2.destroyAllWindows()
                self.camera_manager.release()
                return False

        cv2.destroyAllWindows()
        self.camera_manager.release()
        return calibration_done
                

    
    def _check_calibration_position_quality(self, landmarks) -> bool:
        """
        Check if user is in good position for calibration
        Returns True if all key landmarks are visible with high confidence
        """
        if not landmarks:
            return False
    
    # Check confidence of key landmarks
        key_indices = [11, 12, 13, 14, 23, 24, 25, 26]  # Shoulders, elbows, hips, knees
    
        confidence_threshold = 0.6
        good_count = 0
    
        for idx in key_indices:
            if idx < len(landmarks.landmark):
                if landmarks.landmark[idx].visibility >= confidence_threshold:
                        good_count += 1
    
    # Need at least 6 out of 8 landmarks to be high quality
        return good_count >= 6
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
    def show_countdown(self, duration: int = 3):
        """
        Show visual countdown before exercise starts
        
        Args:
            duration: Countdown duration in seconds
        """
        if not self.camera_manager.is_opened:
            if not self.camera_manager.open():
                return
        
        import time
        for i in range(duration, 0, -1):
            start_time = time.time()
            
            # Show countdown for 1 second
            while time.time() - start_time < 1.0:
                ret, frame = self.camera_manager.read_frame()
                if not ret:
                    continue
                
                # Detect pose for visual feedback
                result = self.pose_detector.detect_pose(frame)
                if result and result.success:
                    frame = self.pose_detector.draw_landmarks(frame, result.landmarks)
                
                # Draw large countdown number
                height, width = frame.shape[:2]
                countdown_text = str(i)
                
                # Huge font for countdown
                (text_width, text_height), _ = cv2.getTextSize(
                    countdown_text, cv2.FONT_HERSHEY_SIMPLEX, 5.0, 10
                )
                x_pos = (width - text_width) // 2
                y_pos = (height + text_height) // 2
                
                # Draw countdown with glow effect
                # Shadow
                self.pose_detector.draw_text(
                    frame, countdown_text, (x_pos + 5, y_pos + 5),
                    font_scale=5.0, color=(0, 0, 0), thickness=12,
                    background=False
                )
                # Main text
                self.pose_detector.draw_text(
                    frame, countdown_text, (x_pos, y_pos),
                    font_scale=5.0, color=(0, 255, 0), thickness=10,
                    background=False
                )
                
                # Instruction text
                instruction = "Get into starting position"
                (inst_width, _), _ = cv2.getTextSize(
                    instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                )
                inst_x = (width - inst_width) // 2
                self.pose_detector.draw_text(
                    frame, instruction, (inst_x, y_pos + 100),
                    font_scale=0.8, color=(255, 255, 255), thickness=2,
                    background=True, background_color=(0, 0, 0)
                )
                
                cv2.imshow("Posture Detection", frame)
                cv2.waitKey(1)
        
        # Final "GO!" message
        for _ in range(5):  # Show for ~0.5 seconds
            ret, frame = self.camera_manager.read_frame()
            if ret:
                result = self.pose_detector.detect_pose(frame)
                if result and result.success:
                    frame = self.pose_detector.draw_landmarks(frame, result.landmarks)
                
                height, width = frame.shape[:2]
                go_text = "GO!"
                (go_width, go_height), _ = cv2.getTextSize(
                    go_text, cv2.FONT_HERSHEY_SIMPLEX, 5.0, 10
                )
                x_pos = (width - go_width) // 2
                y_pos = (height + go_height) // 2
                
                self.pose_detector.draw_text(
                    frame, go_text, (x_pos, y_pos),
                    font_scale=5.0, color=(0, 255, 0), thickness=10,
                    background=False
                )
                
                cv2.imshow("Posture Detection", frame)
                cv2.waitKey(100)
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
                        form_feedback = self.current_exercise.validate_form(measurements)
                        form_score = self.current_exercise.calculate_form_score(measurements)
                        rep_status = self.rep_counter.update(smoothed_angle, form_is_valid=form_feedback.is_correct)
                        
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
                
                if measurements and form_score is not None:
                    frame = self.pose_detector.draw_color_coded_skeleton(
                        frame, 
                        result.landmarks,
                        measurements.angles,
                        self.current_exercise.exercise_name,
                        form_score
                    )
                else:
                    # Fallback to regular skeleton
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
        """
        Draw mobile-optimized feedback layout
        - Top center: Feedback message
        - Bottom center: Rep count & form score
        """
        height, width = frame.shape[:2]
        
        # ===== TOP CENTER: Feedback Message =====
        if feedback_message:
            # Split long messages into multiple lines
            max_chars_per_line = 35
            lines = []
            words = feedback_message.split()
            current_line = ""
            
            for word in words:
                if len(current_line) + len(word) + 1 <= max_chars_per_line:
                    current_line += word + " "
                else:
                    if current_line:
                        lines.append(current_line.strip())
                    current_line = word + " "
            if current_line:
                lines.append(current_line.strip())
            
            # Draw feedback at top center
            y_start = 40
            for i, line in enumerate(lines):
                # Calculate text width for centering
                (text_width, text_height), _ = cv2.getTextSize(
                    line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                x_centered = (width - text_width) // 2
                
                # Color based on form score
                if form_score > 80:
                    color = (0, 255, 0)  # Green - good
                elif form_score > 60:
                    color = (0, 255, 255)  # Yellow - warning
                else:
                    color = (0, 0, 255)  # Red - error
                
                self.pose_detector.draw_text(
                    frame, line, (x_centered, y_start + i * 35),
                    font_scale=0.7, color=color, thickness=2,
                    background=True, background_color=(0, 0, 0)
                )
        
        # ===== BOTTOM CENTER: Rep Count & Stats =====
        bottom_y = height - 100
        
        # Rep count - Large and prominent
        rep_text = f"REPS: {rep_status['rep_count']}"
        (rep_width, rep_height), _ = cv2.getTextSize(
            rep_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3
        )
        rep_x = (width - rep_width) // 2
        
        self.pose_detector.draw_text(
            frame, rep_text, (rep_x, bottom_y),
            font_scale=1.2, color=(255, 255, 255), thickness=3,
            background=True, background_color=(0, 0, 0)
        )
        
        # Form score - Below rep count
        form_text = f"Form: {form_score:.0f}%"
        form_color = (0, 255, 0) if form_score > 80 else (0, 255, 255) if form_score > 60 else (0, 0, 255)
        (form_width, form_height), _ = cv2.getTextSize(
            form_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        )
        form_x = (width - form_width) // 2
        
        self.pose_detector.draw_text(
            frame, form_text, (form_x, bottom_y + 40),
            font_scale=0.8, color=form_color, thickness=2,
            background=True, background_color=(0, 0, 0)
        )
        
        # Phase indicator - Small text at very bottom
        phase_text = f"Phase: {measurements.phase.upper()}"
        (phase_width, _), _ = cv2.getTextSize(
            phase_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        phase_x = (width - phase_width) // 2
        
        self.pose_detector.draw_text(
            frame, phase_text, (phase_x, bottom_y + 70),
            font_scale=0.5, color=(200, 200, 200), thickness=1,
            background=False
        )
        
        return frame

def main():
    """Main entry point"""
    print("=== AI-Based Gym Trainer: Posture Detection System ===\n")
    
    # Initialize system
    system = PostureDetectionSystem(camera_id=1)  # Using laptop camera
    
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
    print("\nGet ready! Exercise will start in 3 seconds...")
    print("- Position your phone/camera")
    print("- Step back into view")
    print("- Get into starting position")
    import time
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    print("START!\n")
    print("Controls: Press 'Q' to quit, 'R' to reset counter")


    system.show_countdown(duration=3)

    
    
    exercise_name = exercise_map.get(choice, 'squat')
    if not system.select_exercise(exercise_name):
        print("Failed to select exercise. Exiting.")
        return
    
    # Run session
    system.run_exercise_session()
    
    print("\nSession completed. Thank you!")


if __name__ == "__main__":
    main()
