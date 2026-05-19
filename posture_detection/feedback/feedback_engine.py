"""
Feedback Engine
Provides real-time text and audio feedback
"""

import time
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class FeedbackEngine:
    """Generate and manage exercise feedback"""
    
    def __init__(self, cooldown_seconds: float = 2.0):
        self.cooldown_seconds = cooldown_seconds
        self.last_feedback_time = {}
        self.last_feedback_message = None
        
    def should_give_feedback(self, feedback_key: str) -> bool:
        """Check if enough time has passed since last feedback"""
        current_time = time.time()
        last_time = self.last_feedback_time.get(feedback_key, 0)
        
        if current_time - last_time >= self.cooldown_seconds:
            self.last_feedback_time[feedback_key] = current_time
            return True
        return False
    
    def generate_feedback(
        self, 
        corrections: List[str],
        rep_count: int = 0,
        form_score: float = 100.0
    ) -> str:
        """Generate feedback message"""
        if not corrections:
            if rep_count > 0:
                return f"Great form! Reps: {rep_count}"
            return "Form looks good!"
        
        feedback_key = '|'.join(sorted(corrections))
        
        if not self.should_give_feedback(feedback_key):
            return self.last_feedback_message or ""
        
        if len(corrections) == 1:
            message = corrections[0]
        else:
            message = f"{len(corrections)} issues: " + corrections[0]
        
        self.last_feedback_message = message
        return message
    
    def format_display(
        self,
        feedback_message: str,
        rep_count: int,
        phase: str,
        form_score: float
    ) -> List[str]:
        """Format feedback for display"""
        lines = []
        lines.append(f"Reps: {rep_count} | Phase: {phase}")
        lines.append(f"Form Score: {form_score:.0f}%")
        if feedback_message:
            lines.append(feedback_message)
        return lines
