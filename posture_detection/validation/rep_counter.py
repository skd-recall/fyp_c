"""
Rep Counter with State Machine
"""

import time
from typing import Optional, Dict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RepState(Enum):
    READY = "ready"
    DOWN = "down"
    UP = "up"


class RepCounter:
    def __init__(self, exercise_name: str, rep_config: Dict):
        self.exercise_name = exercise_name
        self.down_threshold = rep_config.get('down_threshold', 100)
        self.up_threshold = rep_config.get('up_threshold', 165)
        self.min_rep_time = rep_config.get('min_rep_time', 1.0)
        self.mode = rep_config.get('mode', 'reps')
        
        self.current_state = RepState.READY
        self.rep_count = 0
        self.last_rep_time = 0
        self.current_rep_start_time = 0
        self.hold_start_time = None
        self.hold_duration = 0
        self.state_buffer = []
        self.buffer_size = 3
    
    def update(self, angle: float, timestamp: float = None) -> Dict:
        if timestamp is None:
            timestamp = time.time()
        
        self.state_buffer.append(angle)
        if len(self.state_buffer) > self.buffer_size:
            self.state_buffer.pop(0)
        
        if len(self.state_buffer) < self.buffer_size:
            return self._get_status()
        
        import numpy as np
        stable_angle = float(np.median(self.state_buffer))
        
        if self.mode == 'duration':
            is_holding = 155 <= stable_angle <= 185
            if is_holding:
                if self.hold_start_time is None:
                    self.hold_start_time = timestamp
                else:
                    self.hold_duration = timestamp - self.hold_start_time
            else:
                self.hold_start_time = None
                self.hold_duration = 0
        else:
            if self.current_state == RepState.READY:
                if stable_angle <= self.down_threshold:
                    self.current_state = RepState.DOWN
                    self.current_rep_start_time = timestamp
            elif self.current_state == RepState.DOWN:
                if stable_angle >= self.up_threshold:
                    rep_duration = timestamp - self.current_rep_start_time
                    if rep_duration >= self.min_rep_time:
                        self.rep_count += 1
                        self.last_rep_time = timestamp
                        logger.info(f"Rep #{self.rep_count} completed")
                    self.current_state = RepState.READY
        
        return self._get_status()
    
    def _get_status(self) -> Dict:
        return {
            'rep_count': self.rep_count,
            'state': self.current_state.value,
            'mode': self.mode,
            'hold_duration': self.hold_duration if self.mode == 'duration' else 0,
            'is_holding': self.hold_start_time is not None
        }
    
    def reset(self) -> None:
        self.current_state = RepState.READY
        self.rep_count = 0
        self.state_buffer = []
        self.hold_start_time = None
        self.hold_duration = 0
