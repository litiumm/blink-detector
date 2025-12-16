import cv2
import mediapipe as mp
import numpy as np
import time
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils

def eye_aspect_ratio(eye):
    """
    Computes the euclidean distances between the two sets of
    vertical eye landmarks (x, y)-coordinates.
    Then computes the euclidean distance between the horizontal
    eye landmark (x, y)-coordinates.
    Returns the eye aspect ratio.
    """
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def main():
    """
    Main function to run blink and head turn detection.
    """
    # Initialize MediaPipe Holistic
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # --- Constants and Thresholds for Blink Detection ---
    EYE_AR_THRESH = 0.30
    EYE_AR_CONSEC_FRAMES = 3
    # Do not count blinks that are longer than ~2 seconds
    LONG_BLINK_FRAMES = 60
    
    # --- Constants and Thresholds for Head Turn Detection ---
    threshold_x = 20 # X-axis movement threshold
    threshold_y = 20 # Y-axis movement threshold
    STRAIGHT_HEAD_THRESHOLD_X = 10 # Threshold for horizontal nose position
    STRAIGHT_HEAD_MIN_Y_RATIO = 1.8
    STRAIGHT_HEAD_MAX_Y_RATIO = 10.0 
    
    # --- New: Hand near eye detection threshold ---
    HAND_EYE_DIST_THRESH = 40 # Threshold in pixels for hand near eye detection

    # --- Counters and State Variables ---
    COUNTER = 0 # Consecutive frames eye is closed
    TOTAL = 0 # Total blinks
    detection_state = "IDLE" # Can be "IDLE" or "ARMED"
    locked_direction = "" # Stores the first detected direction in a cycle
    
    # --- Head Movement Variables ---
    last_nose_x, last_nose_y = -1, -1 
    
    # Start the video stream
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)

    # Main loop
    while True:
        frame = vs.read()
        if frame is None:
            break

        frame = cv2.flip(frame, 1)
        frame = adjust_gamma(frame, gamma=2.0)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = holistic.process(rgb_frame)
        
        ear = 0.0
        face_detected = False
        is_head_straight = False
        landmarks = None
        is_hand_near_eyes = False

        if results.face_landmarks:
            face_detected = True
            landmarks = results.face_landmarks.landmark
            
            # Use specific MP landmarks for EAR calculation
            left_eye_points = np.array([
                (landmarks[33].x, landmarks[33].y),   # 0
                (landmarks[161].x, landmarks[161].y), # 1
                (landmarks[159].x, landmarks[159].y), # 2
                (landmarks[155].x, landmarks[155].y), # 3
                (landmarks[145].x, landmarks[145].y), # 4
                (landmarks[144].x, landmarks[144].y)  # 5
            ])
            right_eye_points = np.array([
                (landmarks[362].x, landmarks[362].y), # 0
                (landmarks[384].x, landmarks[384].y), # 1
                (landmarks[386].x, landmarks[386].y), # 2
                (landmarks[387].x, landmarks[387].y), # 3
                (landmarks[388].x, landmarks[388].y), # 4
                (landmarks[380].x, landmarks[380].y)  # 5
            ])
            
            # Convert normalized coordinates to pixel coordinates
            h, w, c = frame.shape
            left_eye_points[:, 0] = left_eye_points[:, 0] * w
            left_eye_points[:, 1] = left_eye_points[:, 1] * h
            right_eye_points[:, 0] = right_eye_points[:, 0] * w
            right_eye_points[:, 1] = right_eye_points[:, 1] * h

            leftEAR = eye_aspect_ratio(left_eye_points)
            rightEAR = eye_aspect_ratio(right_eye_points)
            ear = (leftEAR + rightEAR) / 2.0

            # Draw contours for visualization
            leftEyeHull = cv2.convexHull(left_eye_points.astype(np.int32))
            rightEyeHull = cv2.convexHull(right_eye_points.astype(np.int32))
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            
            # Draw the holistic landmarks for visualization
            mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp.solutions.drawing_utils.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp.solutions.drawing_utils.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # --- Hand near eye check ---
            if results.left_hand_landmarks and results.right_hand_landmarks:
                for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
                    for landmark in hand_landmarks.landmark:
                        hand_x = int(landmark.x * w)
                        hand_y = int(landmark.y * h)
                        
                        # Check distance to left eye
                        for eye_point in left_eye_points:
                            eye_x, eye_y = eye_point
                            if dist.euclidean((hand_x, hand_y), (eye_x, eye_y)) < HAND_EYE_DIST_THRESH:
                                is_hand_near_eyes = True
                                break
                        if is_hand_near_eyes:
                            break
                        # Check distance to right eye
                        for eye_point in right_eye_points:
                            eye_x, eye_y = eye_point
                            if dist.euclidean((hand_x, hand_y), (eye_x, eye_y)) < HAND_EYE_DIST_THRESH:
                                is_hand_near_eyes = True
                                break
                    if is_hand_near_eyes:
                        break

            # --- Head Straightness Check ---
            # Horizontal check
            left_inner_eye_x = landmarks[33].x
            right_inner_eye_x = landmarks[362].x
            eye_midpoint_x = (left_inner_eye_x + right_inner_eye_x) / 2.0
            nose_x = landmarks[1].x
            is_horizontally_straight = abs(nose_x - eye_midpoint_x) * w < STRAIGHT_HEAD_THRESHOLD_X
            
            # Vertical check
            nose_y = landmarks[1].y
            chin_y = landmarks[152].y
            eye_y = (landmarks[145].y + landmarks[374].y) / 2.0
            
            if (nose_y - eye_y) > 0:
                vertical_ratio = (chin_y - nose_y) / (nose_y - eye_y)
            else:
                vertical_ratio = 0

            is_vertically_straight = (vertical_ratio > STRAIGHT_HEAD_MIN_Y_RATIO and 
                                      vertical_ratio < STRAIGHT_HEAD_MAX_Y_RATIO)
            
            is_head_straight = is_horizontally_straight and is_vertically_straight

            # --- Blink Detection Logic ---
            # Only track eye state if hand is not near the eyes
            if not is_hand_near_eyes and is_head_straight and ear < EYE_AR_THRESH:
                COUNTER += 1
            else:
                # On eye open, check if a valid blink occurred.
                if is_head_straight and EYE_AR_CONSEC_FRAMES <= COUNTER < LONG_BLINK_FRAMES:
                    TOTAL += 1
                    # Check for two quick blinks
                    if detection_state == "IDLE":
                        if TOTAL % 2 == 0:
                            detection_state = "ARMED"
                            locked_direction = ""
                            last_nose_x, last_nose_y = landmarks[1].x, landmarks[1].y
                
                # Reset the counter regardless of whether a blink was counted
                COUNTER = 0

            # --- Head Turn Detection Logic (only when armed and no direction is locked) ---
            if detection_state == "ARMED" and not locked_direction:
                nose_x = landmarks[1].x
                nose_y = landmarks[1].y
                
                # Convert normalized coordinates to pixel movement
                dx = (nose_x - last_nose_x) * w
                dy = (nose_y - last_nose_y) * h

                if dx < -threshold_x:
                    locked_direction = "Left"
                    detection_state = "IDLE"
                elif dx > threshold_x:
                    locked_direction = "Right"
                    detection_state = "IDLE"
                elif dy > threshold_y:
                    locked_direction = "Down"
                    detection_state = "IDLE"
                elif dy < -threshold_y:
                    locked_direction = "Up"
                    detection_state = "IDLE"
        
        # --- UI Display ---
        cv2.putText(frame, f"Blinks: {TOTAL}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"EAR: {ear:.2f}", (480, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        ui_text = ""
        if detection_state == "IDLE":
            ui_text = "Blink twice to start"
        elif detection_state == "ARMED":
            ui_text = "Turn head..."

        cv2.putText(frame, ui_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        straight_text_color = (0, 255, 0) if is_head_straight else (0, 0, 255)
        straight_text = "Head Straight: Yes" if is_head_straight else "Head Straight: No"
        cv2.putText(frame, straight_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, straight_text_color, 2)
        
        # Display hand near eye status
        if is_hand_near_eyes:
            cv2.putText(frame, "Hand near eyes! Blink detection OFF", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if locked_direction:
            cv2.putText(frame, locked_direction, (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            if face_detected and landmarks:
                (nose_x, nose_y) = (int(landmarks[1].x * w), int(landmarks[1].y * h))
                cv2.circle(frame, (nose_x, nose_y), 3, (0, 255, 0), -1)

        cv2.imshow("Blink and Head Turn Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()
    holistic.close()

if __name__ == "__main__":
    main()
