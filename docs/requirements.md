# Requirements Document

## Introduction

This document specifies the requirements for a Dual-Camera Face Verification System with Deepfake Detection, designed as a BTech final year project. The system uses stereo vision from two cameras to perform liveness detection (anti-spoofing) and face verification, eliminating the need for expensive depth sensors. The system detects presentation attacks (photos, videos, masks) and AI-generated deepfakes by analyzing depth consistency, texture patterns, and temporal features.

### Hardware Configuration

**Primary Setup: Dual Webcam Stereo System**

Two standard USB webcams (720p minimum) mounted with fixed baseline (6-10 cm apart) for stereo depth estimation. This configuration provides:

- 3D facial geometry capture for biometric verification
- Depth-based liveness detection to defeat photo/video attacks
- Cost-effective solution using commodity hardware
- Well-documented stereo vision algorithms (OpenCV)

**Hardware Requirements:**
- 2x USB webcams (720p or higher, matching models preferred)
- Rigid mounting bracket (6-10 cm baseline)
- USB hub or dual USB ports
- Computer with Python support

## Glossary

- **Stereo Vision**: Technique to extract 3D depth information from two 2D camera images by analyzing disparity between corresponding points
- **Disparity Map**: Image showing pixel-wise horizontal displacement between left and right camera views, used to compute depth
- **Liveness Detection**: Process of determining whether a face belongs to a live person or a spoof attempt (photo, video, mask)
- **Deepfake**: AI-generated synthetic media where a person's face is replaced or manipulated using deep learning
- **Presentation Attack**: Attempt to spoof a biometric system using photos, videos, 3D masks, or other artifacts
- **Face Embedding**: Compact numerical vector representation of facial features used for identity comparison
- **Anti-Spoofing**: Techniques to detect and reject fake faces presented to the system
- **Stereo Calibration**: Process of determining intrinsic and extrinsic parameters of a stereo camera setup
- **Rectification**: Image transformation to align stereo image pairs for easier disparity computation
- **ROI (Region of Interest)**: Specific area in an image containing the detected face
- **Baseline**: Distance between the two camera centers in a stereo setup, affects depth accuracy
- **Biometric Template**: Stored face embedding used for identity verification

## Requirements

### Requirement 1: Camera Calibration and Setup

**User Story:** As a system operator, I want to calibrate the camera setup, so that accurate depth or multi-spectral analysis can be performed.

#### Acceptance Criteria

1. WHEN the operator initiates calibration with a checkerboard pattern THEN the Calibration_Module SHALL compute and store intrinsic parameters (focal length, principal point, distortion coefficients) for both cameras
2. WHEN calibration images are captured from both cameras THEN the Calibration_Module SHALL compute extrinsic parameters (rotation matrix, translation vector) between cameras
3. WHEN calibration is complete THEN the Calibration_Module SHALL generate stereo rectification maps for both camera streams
4. WHEN calibration parameters are saved THEN the Calibration_Module SHALL persist parameters to a configuration file in JSON format
5. WHEN loading saved calibration THEN the Calibration_Module SHALL validate parameter integrity before applying rectification

### Requirement 2: Real-Time Stereo Image Acquisition

**User Story:** As a user, I want the system to capture synchronized images from both cameras, so that accurate stereo matching can be performed.

#### Acceptance Criteria

1. WHEN the system starts THEN the Image_Acquisition_Module SHALL initialize both cameras with matching resolution and frame rate settings
2. WHILE cameras are active THEN the Image_Acquisition_Module SHALL capture frames from both cameras with synchronization error below 50 milliseconds
3. WHEN frames are captured THEN the Image_Acquisition_Module SHALL apply rectification using stored calibration parameters
4. IF a camera fails to respond within 100 milliseconds THEN the Image_Acquisition_Module SHALL report the error and attempt reconnection
5. WHEN rectified frames are ready THEN the Image_Acquisition_Module SHALL pass frame pairs to the processing pipeline

### Requirement 3: Face Detection and Tracking

**User Story:** As a user, I want the system to detect and track my face in the camera feed, so that verification can be performed on the correct region.

#### Acceptance Criteria

1. WHEN a frame is received THEN the Face_Detection_Module SHALL detect all faces present using a deep learning detector
2. WHEN multiple faces are detected THEN the Face_Detection_Module SHALL select the largest face closest to frame center for verification
3. WHILE a face is being tracked THEN the Face_Detection_Module SHALL maintain consistent face ID across consecutive frames
4. WHEN a face is detected in the left camera THEN the Face_Detection_Module SHALL locate the corresponding face region in the right camera using stereo geometry
5. IF no face is detected for 30 consecutive frames THEN the Face_Detection_Module SHALL reset tracking state and notify the Verification_Module

### Requirement 4: Depth-Based Liveness Detection

**User Story:** As a security administrator, I want the system to verify that a real 3D face is present, so that photo and video attacks are rejected.

#### Acceptance Criteria

1. WHEN face regions are extracted from both cameras THEN the Depth_Analysis_Module SHALL compute a disparity map for the face ROI
2. WHEN a disparity map is computed THEN the Depth_Analysis_Module SHALL convert disparity values to metric depth using calibration parameters
3. WHEN depth data is available THEN the Depth_Analysis_Module SHALL verify that face depth variation falls within expected range for a 3D human face (nose-to-ear depth between 5-15 cm)
4. IF depth variation is below 2 cm across the face region THEN the Depth_Analysis_Module SHALL classify the input as a flat surface attack
5. WHEN analyzing depth THEN the Depth_Analysis_Module SHALL verify depth continuity without abrupt discontinuities that indicate mask edges
6. WHEN depth analysis is complete THEN the Depth_Analysis_Module SHALL output a liveness confidence score between 0.0 and 1.0

### Requirement 5: Texture-Based Anti-Spoofing

**User Story:** As a security administrator, I want the system to analyze facial texture patterns, so that printed photos and screen replay attacks are detected.

#### Acceptance Criteria

1. WHEN a face ROI is extracted THEN the Texture_Analysis_Module SHALL compute Local Binary Pattern (LBP) histograms from the face region
2. WHEN analyzing texture THEN the Texture_Analysis_Module SHALL detect moiré patterns indicative of screen display attacks
3. WHEN analyzing texture THEN the Texture_Analysis_Module SHALL detect paper texture and printing artifacts from photo attacks
4. WHEN texture features are extracted THEN the Texture_Analysis_Module SHALL classify the input using a trained anti-spoofing model
5. WHEN texture analysis is complete THEN the Texture_Analysis_Module SHALL output a spoof probability score between 0.0 and 1.0



### Requirement 6: Deepfake Detection

**User Story:** As a security administrator, I want the system to detect AI-generated fake faces, so that deepfake attacks are rejected.

#### Acceptance Criteria

1. WHEN a face is detected THEN the Deepfake_Detection_Module SHALL extract facial landmarks and analyze geometric consistency
2. WHEN analyzing for deepfakes THEN the Deepfake_Detection_Module SHALL detect unnatural blending artifacts at face boundaries
3. WHEN analyzing for deepfakes THEN the Deepfake_Detection_Module SHALL analyze eye region for irregular blinking patterns and reflection inconsistencies
4. WHEN analyzing for deepfakes THEN the Deepfake_Detection_Module SHALL detect temporal inconsistencies across consecutive frames
5. WHEN stereo images are available THEN the Deepfake_Detection_Module SHALL verify that facial features maintain consistent depth relationships in both views
6. WHEN deepfake analysis is complete THEN the Deepfake_Detection_Module SHALL output a deepfake probability score between 0.0 and 1.0

### Requirement 7: Face Verification

**User Story:** As a user, I want to verify my identity against enrolled faces, so that I can authenticate myself to the system.

#### Acceptance Criteria

1. WHEN a live face passes anti-spoofing checks THEN the Face_Verification_Module SHALL extract a 512-dimensional face embedding using a deep neural network
2. WHEN an embedding is extracted THEN the Face_Verification_Module SHALL compare the embedding against enrolled templates using cosine similarity
3. WHEN similarity is computed THEN the Face_Verification_Module SHALL accept matches with similarity above 0.6 threshold
4. WHEN verification is complete THEN the Face_Verification_Module SHALL log the verification attempt with timestamp, result, and confidence scores
5. IF all anti-spoofing scores indicate genuine face AND face matches enrolled template THEN the Face_Verification_Module SHALL grant access

### Requirement 8: User Enrollment

**User Story:** As a new user, I want to enroll my face in the system, so that I can use face verification for future authentication.

#### Acceptance Criteria

1. WHEN enrollment is initiated THEN the Enrollment_Module SHALL capture multiple face images from different angles
2. WHEN capturing enrollment images THEN the Enrollment_Module SHALL verify liveness for each captured frame
3. WHEN enrollment images pass liveness checks THEN the Enrollment_Module SHALL extract and average face embeddings from captured images
4. WHEN enrollment is complete THEN the Enrollment_Module SHALL store the face template with associated user identifier
5. WHEN storing enrollment data THEN the Enrollment_Module SHALL encrypt face templates before persistence

### Requirement 9: System Configuration and Thresholds

**User Story:** As a system administrator, I want to configure detection thresholds, so that I can balance security and usability for my deployment.

#### Acceptance Criteria

1. WHEN the system starts THEN the Configuration_Module SHALL load threshold values from a configuration file
2. WHERE custom thresholds are specified THEN the Configuration_Module SHALL apply user-defined values for liveness, spoof, and deepfake detection
3. WHEN thresholds are modified THEN the Configuration_Module SHALL validate that values fall within acceptable ranges
4. WHEN configuration changes are applied THEN the Configuration_Module SHALL log the changes with timestamp and administrator identifier

### Requirement 10: Result Aggregation and Decision

**User Story:** As a user, I want a clear accept/reject decision, so that I know whether my verification attempt succeeded.

#### Acceptance Criteria

1. WHEN all analysis modules complete THEN the Decision_Module SHALL aggregate scores from depth, texture, deepfake, and verification modules
2. WHEN aggregating scores THEN the Decision_Module SHALL apply configurable weights to each score component
3. WHEN a final decision is made THEN the Decision_Module SHALL output ACCEPT, REJECT, or RETRY status
4. IF any anti-spoofing score falls below threshold THEN the Decision_Module SHALL output REJECT with reason code
5. WHEN outputting results THEN the Decision_Module SHALL provide human-readable explanation of the decision

### Requirement 11: Data Serialization and Logging

**User Story:** As a system administrator, I want verification attempts logged, so that I can audit system usage and investigate issues.

#### Acceptance Criteria

1. WHEN a verification attempt completes THEN the Logging_Module SHALL serialize the attempt record to JSON format
2. WHEN serializing records THEN the Logging_Module SHALL include timestamp, user ID, all confidence scores, and final decision
3. WHEN reading stored logs THEN the Logging_Module SHALL deserialize JSON records back to structured objects
4. WHEN logging is enabled THEN the Logging_Module SHALL write records to persistent storage within 100 milliseconds of decision
5. WHEN log files exceed configured size THEN the Logging_Module SHALL rotate to a new file with timestamp suffix
