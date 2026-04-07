"""
FastAPI Backend Endpoints
For integration with React Native frontend
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
import json
from typing import Dict, Optional
import logging

from main import PostureDetectionSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Posture Detection API", version="1.0.0")

# CORS middleware for React Native
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global system instance (one per worker)
posture_system: Optional[PostureDetectionSystem] = None


@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    global posture_system
    posture_system = PostureDetectionSystem(camera_id=1)
    logger.info("Posture Detection System initialized")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Posture Detection API",
        "version": "1.0.0"
    }


@app.get("/exercises")
async def get_available_exercises():
    """Get list of available exercises"""
    return {
        "exercises": [
            {"id": "squat", "name": "Squat", "difficulty": "medium"},
            {"id": "pushup", "name": "Push-up", "difficulty": "easy"},
            {"id": "plank", "name": "Plank", "difficulty": "easy"},
            {"id": "lunge", "name": "Lunge", "difficulty": "medium"},
            {"id": "bicep_curl", "name": "Bicep Curl", "difficulty": "easy"}
        ]
    }


@app.post("/select_exercise")
async def select_exercise(exercise_id: str):
    """Select exercise for detection"""
    global posture_system
    
    if posture_system is None:
        return {"error": "System not initialized"}, 500
    
    success = posture_system.select_exercise(exercise_id)
    
    if success:
        return {
            "success": True,
            "exercise": exercise_id,
            "message": f"Exercise '{exercise_id}' selected"
        }
    else:
        return {
            "success": False,
            "error": "Invalid exercise ID"
        }, 400


@app.post("/process_frame")
async def process_frame(file: UploadFile = File(...)):
    """
    Process single frame
    Used for REST API approach (alternative to WebSocket)
    """
    global posture_system
    
    if posture_system is None:
        return {"error": "System not initialized"}, 500
    
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"error": "Invalid image"}, 400
        
        # Process frame
        result = _process_single_frame(frame)
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        return {"error": str(e)}, 500


@app.websocket("/ws/posture")
async def websocket_posture_detection(websocket: WebSocket):
    """
    WebSocket endpoint for real-time posture detection
    React Native sends frames, receives feedback
    """
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    global posture_system
    
    if posture_system is None:
        await websocket.send_json({"error": "System not initialized"})
        await websocket.close()
        return
    
    try:
        while True:
            # Receive frame data from React Native
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if 'frame' not in message:
                await websocket.send_json({"error": "No frame data"})
                continue
            
            # Decode base64 frame
            frame_b64 = message['frame']
            frame = decode_base64_frame(frame_b64)
            
            if frame is None:
                await websocket.send_json({"error": "Invalid frame"})
                continue
            
            # Process frame
            result = _process_single_frame(frame)
            
            # Send result back to React Native
            await websocket.send_json(result)
    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


def decode_base64_frame(frame_b64: str) -> Optional[np.ndarray]:
    """Decode base64 encoded frame to OpenCV image"""
    try:
        # Remove data URI prefix if present
        if ',' in frame_b64:
            frame_b64 = frame_b64.split(',')[1]
        
        # Decode base64
        frame_bytes = base64.b64decode(frame_b64)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return frame
    except Exception as e:
        logger.error(f"Error decoding frame: {e}")
        return None


def encode_frame_to_base64(frame: np.ndarray) -> str:
    """Encode OpenCV image to base64"""
    try:
        _, buffer = cv2.imencode('.jpg', frame)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{frame_b64}"
    except Exception as e:
        logger.error(f"Error encoding frame: {e}")
        return ""


def _process_single_frame(frame: np.ndarray) -> Dict:
    """Process single frame and return results"""
    global posture_system
    
    if posture_system.current_exercise is None:
        return {"error": "No exercise selected"}
    
    try:
        # Detect pose
        result = posture_system.pose_detector.detect_pose(frame)
        
        if not result or not result.success:
            return {
                "success": False,
                "message": "No pose detected"
            }
        
        # Extract measurements
        measurements = posture_system.current_exercise.extract_measurements(
            result.landmarks, result.frame_width, result.frame_height
        )
        
        if not measurements:
            return {
                "success": False,
                "message": "Failed to extract measurements"
            }
        
        # Smooth angles
        for angle_name, angle_value in measurements.angles.items():
            posture_system.frame_stabilizer.add_angle_measurement(angle_name, angle_value)
        
        # Get primary angle
        primary_angle_key = posture_system._get_primary_angle_key()
        smoothed_angle = posture_system.frame_stabilizer.get_smoothed_angle(primary_angle_key)
        
        if smoothed_angle is None:
            return {"success": False, "message": "Insufficient data"}
        
        # Update rep counter
        rep_status = posture_system.rep_counter.update(smoothed_angle)
        
        # Validate form
        form_feedback = posture_system.current_exercise.validate_form(measurements)
        form_score = posture_system.current_exercise.calculate_form_score(measurements)
        
        # Generate feedback
        feedback_message = posture_system.feedback_engine.generate_feedback(
            form_feedback.corrections,
            rep_status['rep_count'],
            form_score
        )
        
        # Return comprehensive result
        return {
            "success": True,
            "rep_count": rep_status['rep_count'],
            "phase": measurements.phase,
            "form_score": round(form_score, 1),
            "is_correct": form_feedback.is_correct,
            "feedback": feedback_message,
            "corrections": form_feedback.corrections,
            "angles": measurements.angles,
            "state": rep_status['state'],
            "hold_duration": rep_status.get('hold_duration', 0)
        }
        
    except Exception as e:
        logger.error(f"Error in frame processing: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/reset_counter")
async def reset_counter():
    """Reset rep counter"""
    global posture_system
    
    if posture_system and posture_system.rep_counter:
        posture_system.rep_counter.reset()
        return {"success": True, "message": "Counter reset"}
    
    return {"success": False, "error": "System not ready"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
