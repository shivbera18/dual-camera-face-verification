import cv2
import numpy as np
from src.face.detector import FaceResult

def estimate_stereo_depth(left_img: np.ndarray, right_img: np.ndarray, left_face: FaceResult, right_face: FaceResult) -> float:
    """
    Computes a depth disparity map using uncalibrated stereo rectification based on dense 106 facial landmarks.
    Returns the depth variance (standard deviation).
    - A flat surface (iPad, photo) will have extremely low depth variance (e.g. < 1.0)
    - A real 3D human face will have high depth variance (convex profile).
    
    Returns 0.0 if unable to compute depth.
    """
    l_lmk = left_face.dense_landmarks
    r_lmk = right_face.dense_landmarks
    
    if l_lmk is None or r_lmk is None or len(l_lmk) < 8 or len(r_lmk) < 8:
        return 0.0
        
    # 1. Find Fundamental Matrix using 106 dense points
    F, mask = cv2.findFundamentalMat(l_lmk, r_lmk, cv2.FM_RANSAC, 3.0, 0.99)
    if F is None or F.shape != (3, 3):
        return 0.0
        
    h, w = left_img.shape[:2]
    
    # 2. Stereo Rectification (Uncalibrated)
    # This aligns the two cameras so they share the same horizontal epipolar lines
    ret, H1, H2 = cv2.stereoRectifyUncalibrated(l_lmk, r_lmk, F, (w, h))
    if not ret:
        return 0.0
        
    # 3. Warp images to horizontally align them
    left_rectified = cv2.warpPerspective(left_img, H1, (w, h))
    right_rectified = cv2.warpPerspective(right_img, H2, (w, h))
    
    # Convert to grayscale for stereo matching
    left_gray = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
    
    # 4. Compute Disparity Map using Semi-Global Block Matching
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=64,
        blockSize=11,
        P1=8 * 1 * 11**2,
        P2=32 * 1 * 11**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )
    
    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    
    # 5. Extract disparity only inside the face region
    # Warp the left landmarks to match the rectified image
    landmarks_homogeneous = np.hstack([l_lmk, np.ones((len(l_lmk), 1))])
    warped_landmarks = (H1 @ landmarks_homogeneous.T).T
    warped_landmarks = warped_landmarks[:, :2] / warped_landmarks[:, 2:]
    
    x_min = max(0, int(np.min(warped_landmarks[:, 0])))
    y_min = max(0, int(np.min(warped_landmarks[:, 1])))
    x_max = min(w, int(np.max(warped_landmarks[:, 0])))
    y_max = min(h, int(np.max(warped_landmarks[:, 1])))
    
    # Ensure valid bounding box
    if x_max <= x_min or y_max <= y_min:
        return 0.0
        
    face_disparity = disparity[y_min:y_max, x_min:x_max]
    
    # Ignore invalid disparities (-1 or 0)
    valid_disparity = face_disparity[face_disparity > 0]
    
    if len(valid_disparity) < 100:
        return 0.0
        
    # Standard deviation of the depth
    depth_variance = float(np.std(valid_disparity))
    
    # Save a debug image of the disparity map (normalized for viewing)
    try:
        norm_disp = cv2.normalize(face_disparity, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        color_disp = cv2.applyColorMap(norm_disp, cv2.COLORMAP_JET)
        cv2.putText(color_disp, f"Depth Var: {depth_variance:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite("artifacts/metrics/latest_stereo_depth.png", color_disp)
    except:
        pass
        
    return depth_variance
