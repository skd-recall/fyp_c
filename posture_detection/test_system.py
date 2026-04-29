"""
System Test Script
Verify all components are working correctly
"""

import sys


def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    try:
        import cv2
        print("✓ OpenCV installed")
    except ImportError:
        print("✗ OpenCV not found - pip install opencv-python")
        return False
    
    try:
        import mediapipe
        print(f"✓ MediaPipe installed (v{mediapipe.__version__})")
    except ImportError:
        print("✗ MediaPipe not found - pip install mediapipe")
        return False
    
    try:
        import numpy
        print(f"✓ NumPy installed (v{numpy.__version__})")
    except ImportError:
        print("✗ NumPy not found - pip install numpy")
        return False
    
    try:
        import yaml
        print("✓ PyYAML installed")
    except ImportError:
        print("✗ PyYAML not found - pip install pyyaml")
        return False
    
    return True


def test_modules():
    """Test project modules"""
    print("\nTesting project modules...")
    
    try:
        from config.config_manager import config_manager
        print("✓ Config manager")
    except Exception as e:
        print(f"✗ Config manager error: {e}")
        return False
    
    try:
        from core.pose_detector import PoseDetector
        print("✓ Pose detector")
    except Exception as e:
        print(f"✗ Pose detector error: {e}")
        return False
    
    try:
        from exercises.all_exercises import SquatExercise
        print("✓ Exercise implementations")
    except Exception as e:
        print(f"✗ Exercise implementations error: {e}")
        return False
    
    try:
        from validation.rep_counter import RepCounter
        print("✓ Rep counter")
    except Exception as e:
        print(f"✗ Rep counter error: {e}")
        return False
    
    return True


def test_camera():
    """Test camera access"""
    print("\nTesting camera access...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("✗ Camera not accessible (ID: 0)")
            print("  Try: python -c \"import cv2; cap=cv2.VideoCapture(1); print(cap.isOpened())\"")
            cap.release()
            return False
        
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            print(f"✓ Camera working (Resolution: {w}x{h})")
        else:
            print("✗ Camera opened but cannot read frames")
            cap.release()
            return False
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"✗ Camera test error: {e}")
        return False


def test_pose_detection():
    """Test MediaPipe pose detection"""
    print("\nTesting pose detection...")
    
    try:
        from core.pose_detector import PoseDetector
        import cv2
        import numpy as np
        
        detector = PoseDetector()
        
        # Create a dummy frame
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        result = detector.detect_pose(dummy_frame)
        
        if result is not None:
            print("✓ Pose detection working")
            return True
        else:
            print("✗ Pose detection returned None")
            return False
            
    except Exception as e:
        print(f"✗ Pose detection error: {e}")
        return False


def test_config():
    """Test configuration loading"""
    print("\nTesting configuration...")
    
    try:
        from config.config_manager import config_manager
        
        # Test loading exercise config
        squat_config = config_manager.get_exercise_config('squat')
        print(f"✓ Config loaded: {len(squat_config)} keys")
        
        # Test getting exercises
        exercises = config_manager.get_available_exercises()
        print(f"✓ Available exercises: {', '.join(exercises)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("POSTURE DETECTION SYSTEM - COMPONENT TEST")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Modules", test_modules()))
    results.append(("Configuration", test_config()))
    results.append(("Camera", test_camera()))
    results.append(("Pose Detection", test_pose_detection()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20s} {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 All tests passed! System is ready to use.")
        print("\nNext step: Run 'python main.py'")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("- pip install -r requirements.txt")
        print("- Check camera permissions")
        print("- Ensure you're in the project directory")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
