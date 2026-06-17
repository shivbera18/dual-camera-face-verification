from flask import Flask, render_template, Response, jsonify, request
import cv2
import time
import threading

from src.inference.dual_pipeline import build_dual_pipeline
from src.input.dual_camera import DualCameraCapture
from src.enrollment.store import EnrollmentStore

app = Flask(__name__)

pipeline = None
capture = None

# Global state for UI polling
latest_status = {
    "decision": "WAITING",
    "fake_score": 0.0,
    "match_score": 0.0,
    "user_id": "",
    "latency_ms": 0.0
}
current_user = ""

# Enrollment State
enrollment_user = None
enrollment_samples = []

def init_system():
    global pipeline
    if pipeline is None:
        print("Initializing DualCamPipeline...")
        pipeline = build_dual_pipeline(device="auto", checkpoint="artifacts/models/efficientnet_b0_baseline_best.pth")

def generate_frames():
    global latest_status, current_user, capture, enrollment_user, enrollment_samples
    
    if capture is None:
        capture = DualCameraCapture(left_idx=0, right_idx=1, sync_delta_ms=50.0)
        capture.start()
        
    try:
        while True:
            pair = capture.get_pair()
            if pair is None:
                time.sleep(0.01)
                continue
                
            left, right = pair
            
            # 1. Handle Native Enrollment
            if enrollment_user is not None:
                # We use pipeline.single_pipeline.detector to get embeddings silently
                results = pipeline.single_pipeline.detector.detect(left.img)
                if results and len(results) > 0:
                    res = results[0] # Get the most prominent face
                    if res.embedding is not None:
                        if not hasattr(generate_frames, "last_enroll_time"):
                            generate_frames.last_enroll_time = 0
                        if time.time() - generate_frames.last_enroll_time > 0.2:
                            enrollment_samples.append(res.embedding)
                            generate_frames.last_enroll_time = time.time()
                            print(f"Captured {len(enrollment_samples)}/10 for {enrollment_user}")
                        
                latest_status = {
                    "decision": f"ENROLLING {len(enrollment_samples)}/10",
                    "fake_score": 0.0,
                    "match_score": 0.0,
                    "user_id": enrollment_user,
                    "latency_ms": 0.0,
                    "enrollment_done": False
                }
                
                # If we have 10, save it and stop enrolling!
                if len(enrollment_samples) >= 10:
                    try:
                        # Update the active pipeline's memory AND save to disk automatically
                        pipeline.single_pipeline.store.enroll(enrollment_user, enrollment_samples)
                        print(f"Successfully enrolled {enrollment_user}!")
                    except Exception as e:
                        print(f"Failed to save enrollment: {e}")
                    finally:
                        enrollment_user = None
                        enrollment_samples = []
                        latest_status["enrollment_done"] = True
                        
            # 2. Handle Normal Dual-Camera Verification
            elif current_user:
                result = pipeline.run(left, right, current_user)
                latest_status = {
                    "decision": result.decision,
                    "fake_score": result.fake_score,
                    "match_score": result.match_score,
                    "user_id": result.user_id,
                    "latency_ms": result.latency_ms,
                    "enrollment_done": latest_status.get("enrollment_done", False) # Preserve flag
                }
            else:
                latest_status = {
                    "decision": "NO_USER",
                    "fake_score": 0.0,
                    "match_score": 0.0,
                    "user_id": "",
                    "latency_ms": 0.0,
                    "enrollment_done": latest_status.get("enrollment_done", False) # Preserve flag
                }

            # Concatenate left and right cameras for the UI
            view = cv2.hconcat([left.img, right.img])
            
            # Stream the frame to the browser
            ret, buffer = cv2.imencode('.jpg', view)
            if not ret:
                continue
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    except Exception as e:
        print(f"Stream error: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def status():
    return jsonify(latest_status)

@app.route('/api/set_user', methods=['POST'])
def set_user():
    global current_user, enrollment_user
    data = request.json
    current_user = data.get('user_id', '')
    enrollment_user = None # Cancel enrollment if setting user
    return jsonify({"success": True, "current_user": current_user})

@app.route('/api/enroll', methods=['POST'])
def enroll():
    global enrollment_user, enrollment_samples, current_user
    data = request.json
    user_id = data.get('user_id', '')
    if not user_id:
        return jsonify({"success": False, "error": "No user ID provided"})
        
    # Trigger native background enrollment!
    current_user = ""
    enrollment_samples = []
    enrollment_user = user_id
    latest_status["enrollment_done"] = False
    
    return jsonify({"success": True})

if __name__ == '__main__':
    init_system()
    app.run(host='0.0.0.0', port=5050, debug=False, threaded=True)
