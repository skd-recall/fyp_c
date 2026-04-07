"""
Configuration Manager for Posture Detection System
Loads and manages exercise rules and system settings
"""

import yaml
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """Singleton configuration manager for exercise rules"""
    
    _instance = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        config_path = Path(__file__).parent / "exercise_rules.yaml"
        self.load_config(config_path)
        self._initialized = True
    
    
    def load_config(self, config_path: Path) -> None:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise
    
    def get_exercise_config(self, exercise_name: str) -> Dict[str, Any]:
        """Get configuration for specific exercise"""
        exercise_name = exercise_name.lower().replace(" ", "_")
        
        if exercise_name not in self._config:
            available = list(self._config.keys())
            raise ValueError(
                f"Exercise '{exercise_name}' not found in configuration. "
                f"Available: {available}"
            )
        
        return self._config[exercise_name]
    
    def get_general_config(self) -> Dict[str, Any]:
        """Get general system configuration"""
        return self._config.get('general', {})
    
    def get_calibration_config(self) -> Dict[str, Any]:
        """Get calibration configuration"""
        return self._config.get('calibration', {})
    
    def get_confidence_threshold(self) -> float:
        """Get minimum landmark confidence threshold"""
        return self.get_general_config().get('confidence_threshold', 0.7)
    
    def get_smoothing_frames(self) -> int:
        """Get number of frames for smoothing"""
        return self.get_general_config().get('smoothing_frames', 3)
    
    def get_fps_target(self) -> int:
        """Get target FPS"""
        return self.get_general_config().get('fps_target', 30)
    
    def get_feedback_cooldown(self) -> float:
        """Get feedback cooldown in seconds"""
        return self.get_general_config().get('feedback_cooldown', 1.0)
    
    def get_exercise_phases(self, exercise_name: str) -> Dict[str, Any]:
        """Get phases for specific exercise"""
        config = self.get_exercise_config(exercise_name)
        return config.get('phases', {})
    
    def get_rep_counting_config(self, exercise_name: str) -> Dict[str, Any]:
        """Get rep counting configuration for exercise"""
        config = self.get_exercise_config(exercise_name)
        return config.get('rep_counting', {})
    
    def get_form_checks(self, exercise_name: str) -> List[Dict[str, Any]]:
        """Get form check rules for exercise"""
        config = self.get_exercise_config(exercise_name)
        return config.get('form_checks', [])
    
    def get_key_landmarks(self, exercise_name: str) -> List[str]:
        """Get key landmarks for exercise"""
        config = self.get_exercise_config(exercise_name)
        return config.get('key_landmarks', [])
    
    def get_camera_view(self, exercise_name: str) -> str:
        """Get recommended camera view for exercise"""
        config = self.get_exercise_config(exercise_name)
        return config.get('camera_view', 'side')
    
    def is_calibration_required(self) -> bool:
        """Check if calibration is required"""
        calib_config = self.get_calibration_config()
        return calib_config.get('required', True)
    
    def get_available_exercises(self) -> List[str]:
        """Get list of all available exercises"""
        # Filter out 'general' and 'calibration' keys
        return [
            key for key in self._config.keys() 
            if key not in ['general', 'calibration']
        ]
    
    def validate_angle_range(
        self, 
        angle: float, 
        range_key: str, 
        exercise_name: str, 
        phase: str,
        tolerance: float = 5.0
    ) -> bool:
        """
        Validate if angle falls within acceptable range
        
        Args:
            angle: Measured angle in degrees
            range_key: Key for angle range (e.g., 'knee_angle')
            exercise_name: Name of exercise
            phase: Current phase of exercise
            tolerance: Additional tolerance in degrees (default ±5°)
        
        Returns:
            True if angle is valid, False otherwise
        """
        phases = self.get_exercise_phases(exercise_name)
        
        if phase not in phases:
            logger.warning(f"Phase '{phase}' not found for {exercise_name}")
            return False
        
        phase_config = phases[phase]
        
        if range_key not in phase_config:
            logger.warning(f"Range key '{range_key}' not found in {phase} phase")
            return False
        
        angle_range = phase_config[range_key]
        min_angle = angle_range[0] - tolerance
        max_angle = angle_range[1] + tolerance
        
        return min_angle <= angle <= max_angle
    
    def get_angle_range_with_tolerance(
        self,
        exercise_name: str,
        phase: str,
        range_key: str,
        tolerance: float = 5.0
    ) -> tuple:
        """Get angle range with tolerance applied"""
        phases = self.get_exercise_phases(exercise_name)
        phase_config = phases.get(phase, {})
        angle_range = phase_config.get(range_key, [0, 180])
        
        return (
            angle_range[0] - tolerance,
            angle_range[1] + tolerance
        )


# Global instance
config_manager = ConfigManager()
