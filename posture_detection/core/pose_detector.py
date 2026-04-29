"""
MediaPipe Pose Detector
Wrapper for MediaPipe Pose with optimized settings for fitness tracking
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple
import logging
from dataclasses import dataclass
from utils.landmark_utils import LandmarkExtractor

logger = logging.getLogger(__name__)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


@dataclass
class PoseDetectionResult:
    """Results from pose detection"""
    landmarks: any  # MediaPipe pose landmarks
    world_landmarks: any  # World coordinates (normalized)
    success: bool
    frame_width: int
    frame_height: int


class PoseDetector:
    """
    MediaPipe Pose Detection wrapper
    Optimized for real-time fitness tracking
    """
    
    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        enable_segmentation: bool = False,
        smooth_segmentation: bool = True,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7
    ):
        """
        Initialize pose detector
        
        Args:
            static_image_mode: If False, treats input as video stream
            model_complexity: 0, 1, or 2 (higher = more accurate but slower)
            smooth_landmarks: Whether to smooth landmarks across frames
            enable_segmentation: Whether to generate segmentation mask
            smooth_segmentation: Whether to smooth segmentation mask
            min_detection_confidence: Minimum confidence for person detection
            min_tracking_confidence: Minimum confidence for landmark tracking
        """
        
        self.pose = mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            smooth_segmentation=smooth_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.is_initialized = True
        logger.info(f"PoseDetector initialized with complexity={model_complexity}")
    
    def detect_pose(self, frame: np.ndarray) -> Optional[PoseDetectionResult]:
        """
        Detect pose in frame
        
        Args:
            frame: BGR image from OpenCV
        
        Returns:
            PoseDetectionResult or None if detection fails
        """
        if not self.is_initialized:
            logger.error("Pose detector not initialized")
            return None
        
        try:
            # Convert BGR to RGB (MediaPipe expects RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Make frame writeable=False for performance
            frame_rgb.flags.writeable = False
            
            # Process frame
            results = self.pose.process(frame_rgb)
            
            # Make frame writeable again
            frame_rgb.flags.writeable = True
            
            # Get frame dimensions
            height, width = frame.shape[:2]
            
            if results.pose_landmarks:
                return PoseDetectionResult(
                    landmarks=results.pose_landmarks,
                    world_landmarks=results.pose_world_landmarks,
                    success=True,
                    frame_width=width,
                    frame_height=height
                )
            else:
                logger.debug("No pose detected in frame")
                return PoseDetectionResult(
                    landmarks=None,
                    world_landmarks=None,
                    success=False,
                    frame_width=width,
                    frame_height=height
                )
        
        except Exception as e:
            logger.error(f"Error in pose detection: {e}")
            return None
    
    def draw_landmarks(
        self, 
        frame: np.ndarray, 
        landmarks,
        draw_body: bool = True,
        draw_face: bool = False,
        draw_hands: bool = False
    ) -> np.ndarray:
        """
        Draw pose landmarks on frame
        Hide face landmarks for privacy/clarity
        
        Args:
            frame: Image to draw on
            landmarks: MediaPipe pose landmarks
            draw_body: Draw body skeleton
           
            draw_hands: Draw hand landmarks
        
        Returns:
            Frame with landmarks drawn
        """
        if landmarks is None:
            return frame
        
        try:
            body_connections = [
                # Torso
                (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
                (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
                (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
                (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
            
                # Left arm
                (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
                (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
            
                # Right arm
                (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
                (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
            
                # Left leg
                (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
                (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
            
                # Right leg
                (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
                (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
            ]
        
            if draw_body:
                # Draw only body landmarks (no face)
                mp_drawing.draw_landmarks(
                    frame,
                    landmarks,
                    body_connections,  # Use body-only connections
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0, 255, 0),  # Green joints
                        thickness=2,
                        circle_radius=3
                    ),
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(255, 255, 255),  # White lines
                        thickness=2
                    )
                )     
                
                
            
            return frame
            
        except Exception as e:
            logger.error(f"Error drawing landmarks: {e}")
            return frame
    
    def draw_custom_landmarks(
        self,
        frame: np.ndarray,
        landmarks_dict: dict,
        connections: list = None,
        landmark_color: Tuple[int, int, int] = (0, 255, 0),
        connection_color: Tuple[int, int, int] = (255, 0, 0),
        landmark_radius: int = 5,
        connection_thickness: int = 2
    ) -> np.ndarray:
        """
        Draw custom landmarks and connections
        
        Args:
            frame: Image to draw on
            landmarks_dict: Dictionary of landmark names to (x, y) coordinates
            connections: List of tuples (landmark1_name, landmark2_name)
            landmark_color: Color for landmark points (B, G, R)
            connection_color: Color for connections (B, G, R)
            landmark_radius: Radius of landmark circles
            connection_thickness: Thickness of connection lines
        
        Returns:
            Frame with custom landmarks drawn
        """
        try:
            # Draw connections first (so they appear under landmarks)
            if connections:
                for connection in connections:
                    landmark1, landmark2 = connection
                    if landmark1 in landmarks_dict and landmark2 in landmarks_dict:
                        pt1 = (int(landmarks_dict[landmark1][0]), 
                               int(landmarks_dict[landmark1][1]))
                        pt2 = (int(landmarks_dict[landmark2][0]), 
                               int(landmarks_dict[landmark2][1]))
                        cv2.line(frame, pt1, pt2, connection_color, connection_thickness)
            
            # Draw landmarks
            for landmark_name, (x, y) in landmarks_dict.items():
                cv2.circle(frame, (int(x), int(y)), landmark_radius, landmark_color, -1)
                # Optionally draw landmark name
                # cv2.putText(frame, landmark_name, (int(x), int(y)-10), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error drawing custom landmarks: {e}")
            return frame
    
    def calculate_fps(self, frame_count: int, start_time: float, current_time: float) -> float:
        """
        Calculate current FPS
        
        Args:
            frame_count: Number of frames processed
            start_time: Start time in seconds
            current_time: Current time in seconds
        
        Returns:
            Frames per second
        """
        elapsed_time = current_time - start_time
        if elapsed_time == 0:
            return 0.0
        return frame_count / elapsed_time
    def draw_color_coded_skeleton(
        self,
        frame: np.ndarray,
        landmarks,
        angles: dict,
        exercise_name: str,
        form_score: float
    ) -> np.ndarray:
        """
        Draw skeleton with color-coded joints based on form quality
        Green = good angle, Yellow = acceptable, Red = bad angle
        """
        if landmarks is None:
            return frame
        
        height, width = frame.shape[:2]
        
        # Define joints to check for each exercise
        key_joints = {
            'squat': ['LEFT_KNEE', 'LEFT_HIP'],
            'shoulder_press': ['LEFT_ELBOW', 'LEFT_SHOULDER'],
            'pushup': ['LEFT_ELBOW', 'LEFT_SHOULDER'],
            'plank': ['LEFT_HIP'],
            'bicep_curl': ['LEFT_ELBOW']
        }
        
        important_joints = key_joints.get(exercise_name, [])
        
        # Draw connections
        connections = [
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
            (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
            (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
            (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
            (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
            (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
            (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
            (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
            (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
            (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
        ]
        
        # Overall form color
        if form_score > 80:
            connection_color = (0, 255, 0)  # Green
        elif form_score > 60:
            connection_color = (0, 255, 255)  # Yellow
        else:
            connection_color = (0, 0, 255)  # Red
        
        # Draw connections
        for connection in connections:
            start_idx = connection[0].value
            end_idx = connection[1].value
            
            if start_idx < len(landmarks.landmark) and end_idx < len(landmarks.landmark):
                start = landmarks.landmark[start_idx]
                end = landmarks.landmark[end_idx]
                
                if start.visibility > 0.5 and end.visibility > 0.5:
                    start_point = (int(start.x * width), int(start.y * height))
                    end_point = (int(end.x * width), int(end.y * height))
                    
                    cv2.line(frame, start_point, end_point, connection_color, 3)
        
        # Draw joints with specific colors
        for idx, landmark in enumerate(landmarks.landmark):
            if landmark.visibility < 0.5:
                continue
            
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            
            # Determine joint color
            landmark_name = None
            for name, lm_idx in LandmarkExtractor.LANDMARK_MAP.items():
                if lm_idx.value == idx:
                    landmark_name = name
                    break
            
            if landmark_name in important_joints:
                # Key joint - color based on form
                joint_color = connection_color
                radius = 8
            else:
                # Regular joint
                joint_color = (255, 255, 255)  # White
                radius = 5
            
            cv2.circle(frame, (x, y), radius, joint_color, -1)
            cv2.circle(frame, (x, y), radius + 2, (0, 0, 0), 2)  # Black outline
        
        return frame
    def draw_fps(
        self, 
        frame: np.ndarray, 
        fps: float,
        position: Tuple[int, int] = (10, 30)
    ) -> np.ndarray:
        """
        Draw FPS counter on frame
        
        Args:
            frame: Image to draw on
            fps: FPS value to display
            position: (x, y) position for text
        
        Returns:
            Frame with FPS displayed
        """
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(
            frame, 
            fps_text, 
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        return frame
    
    def draw_text(
        self,
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        font_scale: float = 0.7,
        color: Tuple[int, int, int] = (255, 255, 255),
        thickness: int = 2,
        background: bool = True,
        background_color: Tuple[int, int, int] = (0, 0, 0)
    ) -> np.ndarray:
        """
        Draw text with optional background
        
        Args:
            frame: Image to draw on
            text: Text to display
            position: (x, y) position
            font_scale: Size of font
            color: Text color (B, G, R)
            thickness: Text thickness
            background: Whether to draw background rectangle
            background_color: Background color (B, G, R)
        
        Returns:
            Frame with text drawn
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        if background:
            # Get text size to draw background
            (text_width, text_height), baseline = cv2.getTextSize(
                text, font, font_scale, thickness
            )
            
            # Draw background rectangle
            x, y = position
            cv2.rectangle(
                frame,
                (x, y - text_height - baseline),
                (x + text_width, y + baseline),
                background_color,
                -1
            )
        
        # Draw text
        cv2.putText(
            frame,
            text,
            position,
            font,
            font_scale,
            color,
            thickness
        )
        
        return frame
    
    def draw_angle(
        self,
        frame: np.ndarray,
        angle: float,
        vertex_point: Tuple[int, int],
        angle_name: str = "",
        color: Tuple[int, int, int] = (255, 255, 0)
    ) -> np.ndarray:
        """
        Draw angle measurement at vertex point
        
        Args:
            frame: Image to draw on
            angle: Angle value in degrees
            vertex_point: (x, y) position of vertex
            angle_name: Optional name for angle
            color: Text color (B, G, R)
        
        Returns:
            Frame with angle displayed
        """
        text = f"{angle:.1f}°"
        if angle_name:
            text = f"{angle_name}: {text}"
        
        # Offset text slightly from vertex
        position = (vertex_point[0] + 10, vertex_point[1] - 10)
        
        return self.draw_text(frame, text, position, color=color, font_scale=0.5)
    
    def release(self) -> None:
        """Release MediaPipe resources"""
        if self.is_initialized:
            self.pose.close()
            self.is_initialized = False
            logger.info("Pose detector released")
    
    def __del__(self):
        """Destructor to ensure resources are released"""
        self.release()


class CameraManager:
    """Manage camera input for pose detection"""
    
    def __init__(self, camera_id: int = 1, width: int = 640, height: int = 480):
        """
        Initialize camera
        
        Args:
            camera_id: Camera device ID (1 for default)
            width: Frame width
            height: Frame height
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap = None
        self.is_opened = False
        
        logger.info(f"Camera manager initialized (id={camera_id}, {width}x{height})")
    
    def open(self) -> bool:
        """
        Open camera
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_id}")
                return False
            
            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # Set FPS (if supported)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_opened = True
            logger.info(f"Camera {self.camera_id} opened successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error opening camera: {e}")
            return False
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read frame from camera
        
        Returns:
            Tuple of (success, frame)
        """
        if not self.is_opened:
            return False, None
        
        ret, frame = self.cap.read()
        return ret, frame
    
    def release(self) -> None:
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False
            logger.info("Camera released")
    
    def __del__(self):
        """Destructor to ensure camera is released"""
        self.release()
