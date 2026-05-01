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
        self.form_valid_during_rep = False
        self.previous_angle = None
        self.direction = 'down'  # 'up' = curling, 'down' = lowering
        self.state_buffer = []
        self.buffer_size = 3
    def update(self, angle: float, form_is_valid: bool = True, timestamp: float = None) -> Dict:
        if timestamp is None:
            timestamp = time.time()

        self.state_buffer.append(angle)
        if len(self.state_buffer) > self.buffer_size:
            self.state_buffer.pop(0)

        if len(self.state_buffer) < self.buffer_size:
            return self._get_status()

        import numpy as np
        stable_angle = float(np.median(self.state_buffer))
        # Direction tracking - which way is the angle moving?
        if self.previous_angle is not None:
            if stable_angle < self.previous_angle - 1.0:
                self.direction = 'up'    # angle decreasing = curling up
            elif stable_angle > self.previous_angle + 1.0:
                self.direction = 'down'  # angle increasing = lowering
            # else: no direction change (stable), keep previous direction
        self.previous_angle = stable_angle

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
                    self.form_valid_during_rep = form_is_valid  # start tracking form

            elif self.current_state == RepState.DOWN:
                # Track if form stayed valid during the curl
                if form_is_valid:
                    self.form_valid_during_rep = True

                if stable_angle >= self.up_threshold:
                    rep_duration = timestamp - self.current_rep_start_time
                    # Only count if: enough time passed AND form was valid at least once
                    if rep_duration >= self.min_rep_time and self.form_valid_during_rep:
                        self.rep_count += 1
                        self.last_rep_time = timestamp
                        logger.info(f"Rep #{self.rep_count} completed (valid form)")
                    elif not self.form_valid_during_rep:
                        logger.info("Rep NOT counted - form was incorrect throughout")
                    self.current_state = RepState.READY
                    self.form_valid_during_rep = False

        return self._get_status()
    
    def _get_status(self) -> Dict:
        return {
            'rep_count': self.rep_count,
            'state': self.current_state.value,
            'mode': self.mode,
            'hold_duration': self.hold_duration if self.mode == 'duration' else 0,
            'is_holding': self.hold_start_time is not None
            'direction': self.direction
        }
    
    def reset(self) -> None:
        self.current_state = RepState.READY
        self.rep_count = 0
        self.state_buffer = []
        self.hold_start_time = None
        self.hold_duration = 0
        self.form_valid_during_rep = False