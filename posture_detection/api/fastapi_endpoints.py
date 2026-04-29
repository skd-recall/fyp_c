"""
FastAPI WebSocket Backend
Matches React Native frontend protocol exactly
Protocol: waiting → calibrating → countdown → detecting → completed
"""
from uvicorn.protocols.utils import ClientDisconnected
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import base64
import json
import asyncio
import logging
from typing import Optional
import sys
import os
import asyncio
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import PostureDetectionSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Gym Trainer API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Health check ──────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "online", "service": "AI Gym Trainer API"}


@app.get("/exercises")
async def get_exercises():
    return {
        "exercises": [
            {"id": "squat",          "name": "Squat"},
            {"id": "pushup",         "name": "Push Up"},
            {"id": "plank",          "name": "Plank"},
            {"id": "lunge",          "name": "Lunge"},
            {"id": "shoulder_press", "name": "Shoulder Press"},
            {"id": "bicep_curl",     "name": "Bicep Curl"},
        ]
    }


# ─── Helper: decode base64 frame ──────────────────────────────
def decode_frame(frame_b64: str) -> Optional[np.ndarray]:
    try:
        import cv2
        if ',' in frame_b64:
            frame_b64 = frame_b64.split(',')[1]
        frame_bytes = base64.b64decode(frame_b64)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        logger.error(f"Frame decode error: {e}")
        return None


# ─── Helper: extract skeleton points ──────────────────────────
def extract_skeleton(landmarks) -> list:
    if not landmarks:
        return []
    return [
        {
            "id": idx,
            "x": lm.x,
            "y": lm.y,
            "z": lm.z,
            "visibility": lm.visibility
        }
        for idx, lm in enumerate(landmarks.landmark)
    ]


# ─── Main WebSocket endpoint ───────────────────────────────────
@app.websocket("/ws/posture")
async def websocket_posture(websocket: WebSocket):
    await websocket.accept()
    logger.info("✅ New WebSocket connection")

    # Each connection gets its own system instance
    system = PostureDetectionSystem(camera_id=0)

    # Session state
    session_phase = "idle"
    # idle → waiting → calibrating → countdown → detecting → completed

    calibration_frames = 0
    CALIBRATION_FRAMES_NEEDED = 5 # ~3 seconds at 10fps

    countdown_value = 3
    countdown_frames = 0
    COUNTDOWN_FRAMES_PER_COUNT = 10  # 10 frames = 1 second at 10fps

    exercise_selected = False
    total_form_scores = []

    try:
        while True:
            # Receive message from React Native
            try:
                raw = await asyncio.wait_for(
                websocket.receive_text(),
                timeout=30.0
                )
            except asyncio.TimeoutError:
                logger.info("⏰ Connection timeout")
                break
            message = json.loads(raw)
            action = message.get("action", "")

            # ── Handle: select_exercise ──────────────────────
            if action == "select_exercise":
                exercise_id = message.get("exercise", "squat")
                success = system.select_exercise(exercise_id)

                if success:
                    exercise_selected = True
                    session_phase = "waiting"
                    calibration_frames = 0
                    countdown_value = 3
                    countdown_frames = 0
                    total_form_scores = []

                    logger.info(f"✅ Exercise selected: {exercise_id}")
                    await websocket.send_json({
                        "status": "waiting",
                        "message": "Please step into frame"
                    })
                else:
                    await websocket.send_json({
                        "status": "error",
                        "message": f"Unknown exercise: {exercise_id}"
                    })
                continue

            # ── Handle: stop_session ─────────────────────────
            if action == "stop_session":
                avg_score = (
                    sum(total_form_scores) / len(total_form_scores)
                    if total_form_scores else 0
                )
                await websocket.send_json({
                    "status": "completed",
                    "total_reps": system.rep_counter.rep_count if system.rep_counter else 0,
                    "avg_form_score": round(avg_score, 1)
                })
                logger.info("🛑 Session stopped by user")
                break

            # ── Handle: process_frame ────────────────────────
            if action == "process_frame":
                logger.info("📸 Frame received")
                await websocket.send_json({
                    "status": "waiting",
                    "message": "Processing..."
                })
                if not exercise_selected:
                    await websocket.send_json({
                        "status": "error",
                        "message": "No exercise selected"
                    })
                    continue

                frame_b64 = message.get("frame", "")
                frame = decode_frame(frame_b64)

                if frame is None:

                  
                    continue

                # Run pose detection
                # Run pose detection in thread pool (non-blocking)
                t1 = time.time()
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, system.pose_detector.detect_pose, frame
                )
                t2 = time.time()
                logger.info(f"⏱️ Pose detection took: {t2-t1:.2f}s")

                # ── No person in frame ───────────────────────
                if not result or not result.success:
                    session_phase = "waiting"
                    calibration_frames = 0
                    await websocket.send_json({
                        "status": "waiting",
                        "message": "Please step into frame"
                    })
                    continue

                # Person detected — check quality
# Person detected — check quality (in thread pool)
                is_good, quality, quality_msg = await loop.run_in_executor(
                    None,
                    system.calibrator.check_frame_quality,
                    result.landmarks,
                    result.frame_width,
                    result.frame_height
                )
                logger.info(f"👁️ Frame quality: {quality:.0f}% good={is_good} phase={session_phase}")

                # ── WAITING → CALIBRATING ────────────────────
                if session_phase == "waiting":
                    if is_good:
                        session_phase = "calibrating"
                        calibration_frames = 0
                        try:
                            await websocket.send_json({
                                "status": "calibrating",
                                "message": "Hold still, calibrating..."
                            })
                            logger.info("📤 Sent calibrating start")
                        except Exception as send_err:
                            logger.error(f"❌ Send failed: {send_err}")
                            break
                        
                    else:
                        try:
                            await websocket.send_json({
                                "status": "waiting",
                                "message": quality_msg or "Step back so full body is visible"
                            })
                            logger.info("📤 Sent waiting response")
                        except Exception as send_err:
                            logger.error(f"❌ Send failed: {send_err}")
                            break
                       
                    continue

                # ── CALIBRATING ──────────────────────────────
                if session_phase == "calibrating":
                    if not is_good:
                        # User moved out of frame during calibration
                        session_phase = "waiting"
                        calibration_frames = 0
                        await websocket.send_json({
                            "status": "waiting",
                            "message": "Please step into frame again"
                        })
                        continue

                    calibration_frames += 1
                    remaining = max(0, CALIBRATION_FRAMES_NEEDED - calibration_frames)

                    if calibration_frames >= CALIBRATION_FRAMES_NEEDED:
                        # Try to calibrate
                        success = await loop.run_in_executor(
                            None,
                            system.calibrator.calibrate_from_frame,
                            result.landmarks,
                            result.frame_width,
                            result.frame_height
                        )
                        logger.info(f"🔧 Calibration result: {success}")

                        if success:
                            system.is_calibrated = True
                            # Give calibrator to exercise
                            if system.current_exercise:
                                system.current_exercise.calibrator = system.calibrator

                            session_phase = "countdown"
                            countdown_value = 3
                            countdown_frames = 0
                            logger.info("✅ Calibration complete")

                            await websocket.send_json({
                                "status": "countdown",
                                "count": countdown_value
                            })
                        else:
                            # Calibration failed - restart
                            session_phase = "waiting"
                            calibration_frames = 0
                            await websocket.send_json({
                                "status": "waiting",
                                "message": "Calibration failed. Step back and try again."
                            })
                    else:
                        try:
                            await websocket.send_json({
                                "status": "calibrating",
                                "message": f"Hold still... {remaining // 10 + 1}s"
                            })
                            logger.info("📤 Sent calibrating response")
                        except Exception as send_err:
                            logger.error(f"❌ Send failed: {send_err}")
                            break
                    continue

                # ── COUNTDOWN ────────────────────────────────
                if session_phase == "countdown":
                    countdown_frames += 1

                    if countdown_frames >= COUNTDOWN_FRAMES_PER_COUNT:
                        countdown_frames = 0
                        countdown_value -= 1

                        if countdown_value <= 0:
                            # Countdown done - start detecting
                            session_phase = "detecting"

                            # Initialize rep counter
                            system.rep_counter.reset()
                            system.frame_stabilizer.reset()
                            logger.info("🏋️ Detection started!")

                            await websocket.send_json({
                                "status": "detecting",
                                "rep_count": 0,
                                "form_score": 0,
                                "phase": "",
                                "feedback": "Go!",
                                "corrections": [],
                                "skeleton_points": extract_skeleton(result.landmarks),
                                "angles": {}
                            })
                        else:
                            await websocket.send_json({
                                "status": "countdown",
                                "count": countdown_value
                            })
                    else:
                        # Still in same count
                        await websocket.send_json({
                            "status": "countdown",
                            "count": countdown_value
                        })
                    continue

                # ── DETECTING ────────────────────────────────
                if session_phase == "detecting":
                    try:
                        # Extract measurements
                        measurements = system.current_exercise.extract_measurements(
                            result.landmarks,
                            result.frame_width,
                            result.frame_height
                        )

                        if not measurements:
                            await websocket.send_json({
                                "status": "detecting",
                                "rep_count": system.rep_counter.rep_count if system.rep_counter else 0,
                                "form_score": 0,
                                "phase": "",
                                "feedback": "Stay in frame",
                                "corrections": [],
                                "skeleton_points": extract_skeleton(result.landmarks),
                                "angles": {}
                            })
                            continue

                        # Smooth angles
                        for angle_name, angle_value in measurements.angles.items():
                            system.frame_stabilizer.add_angle_measurement(
                                angle_name, angle_value
                            )

                        # Get primary angle for rep counting
                        primary_key = system._get_primary_angle_key()
                        smoothed_angle = system.frame_stabilizer.get_smoothed_angle(primary_key)

                        # Count reps
                        rep_status = {"rep_count": 0, "state": "unknown"}
                        if smoothed_angle and system.rep_counter:
                            rep_status = system.rep_counter.update(smoothed_angle)

                        # Validate form
                        form_feedback = system.current_exercise.validate_form(measurements)
                        form_score = system.current_exercise.calculate_form_score(measurements)

                        # Generate feedback message
                        feedback_message = system.feedback_engine.generate_feedback(
                            form_feedback.corrections,
                            rep_status.get("rep_count", 0),
                            form_score
                        )

                        # Track form scores
                        if form_score > 0:
                            total_form_scores.append(form_score)

                        # Build skeleton
                        skeleton_points = extract_skeleton(result.landmarks)

                        await websocket.send_json({
                            "status": "detecting",
                            "rep_count": rep_status.get("rep_count", 0),
                            "form_score": round(form_score, 1),
                            "phase": measurements.phase,
                            "feedback": feedback_message,
                            "corrections": form_feedback.corrections,
                            "skeleton_points": skeleton_points,
                            "angles": measurements.angles,
                            "exercise_state": rep_status.get("state", "")
                        })

                    except Exception as e:
                        logger.error(f"Detection error: {e}")
                        await websocket.send_json({
                            "status": "detecting",
                            "rep_count": system.rep_counter.rep_count if system.rep_counter else 0,
                            "form_score": 0,
                            "phase": "",
                            "feedback": "",
                            "corrections": [],
                            "skeleton_points": [],
                            "angles": {}
                        })
                    continue

    except (WebSocketDisconnect,ClientDisconnected):
        logger.info("📱 Client disconnected")
    except Exception as e:
        import traceback
        logger.error(f"💥 CRASH: {type(e).__name__}: {e}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        try:
            await websocket.send_json({
                "status": "error",
                "message": str(e)
            })
        except:
            pass
    finally:
        logger.info("🔌 WebSocket connection closed")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")