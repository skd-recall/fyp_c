# Quick Setup Guide

## Installation (5 minutes)

### Step 1: Install Dependencies
```bash
cd posture_detection
pip install -r requirements.txt
```

### Step 2: Test Installation
```bash
python -c "import cv2, mediapipe; print('✓ All dependencies installed successfully!')"
```

### Step 3: Run the System
```bash
python main.py
```

## First-Time Workflow

1. **Calibration** (3 seconds)
   - Stand facing camera
   - Feet shoulder-width apart
   - Arms at sides
   - Hold steady

2. **Select Exercise**
   - Choose from 5 exercises
   - System loads exercise-specific rules

3. **Start Workout**
   - Stand in camera view
   - Begin exercise
   - Watch real-time feedback
   - System counts reps automatically

## Controls

- **Q**: Quit session
- **R**: Reset rep counter

## System Check

```bash
# Verify camera access
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera:', cap.isOpened()); cap.release()"

# Check MediaPipe
python -c "import mediapipe as mp; print('MediaPipe version:', mp.__version__)"
```

## Common Issues

### Camera Not Found
```bash
# Try different camera IDs
python main.py  # Uses camera 0 by default
# Edit main.py: PostureDetectionSystem(camera_id=1)
```

### Low FPS
- Close other applications using camera
- Reduce resolution in `core/pose_detector.py`
- Use `model_complexity=0` for faster (but less accurate) detection

### "Module not found" Error
```bash
# Ensure you're in the project directory
cd posture_detection
# Reinstall dependencies
pip install -r requirements.txt
```

## Integration with Your App

### For React Native + FastAPI:

1. **Start FastAPI Backend**:
```bash
cd api
python fastapi_endpoints.py
# Server runs on http://localhost:8000
```

2. **Connect from React Native**:
```javascript
const ws = new WebSocket('ws://YOUR_SERVER_IP:8000/ws/posture');

// Send frame
ws.send(JSON.stringify({
  frame: frameBase64
}));

// Receive feedback
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Reps:', data.rep_count);
  console.log('Form Score:', data.form_score);
};
```

## File Structure Reference

```
posture_detection/
├── main.py                    ← Start here
├── config/
│   └── exercise_rules.yaml    ← Adjust thresholds here
├── exercises/
│   └── all_exercises.py       ← Add new exercises here
└── api/
    └── fastapi_endpoints.py   ← Backend integration
```

## Next Steps

1. ✅ Run `python main.py` to test
2. 📝 Review `README.md` for detailed documentation
3. 🔧 Customize thresholds in `exercise_rules.yaml` if needed
4. 🚀 Integrate with your React Native app using FastAPI endpoints

## Support

- Check `README.md` for comprehensive documentation
- Review exercise implementations in `exercises/all_exercises.py`
- Adjust camera settings in `core/pose_detector.py`

**You're all set! Happy coding! 🚀**
