import cv2
import mediapipe as mp
import numpy as np
import time
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from gpiozero import DigitalOutputDevice, PWMOutputDevice

# ---------------- MOTOR SETUP (L298N) ----------------
in1 = DigitalOutputDevice(17)
in2 = DigitalOutputDevice(18)
in3 = DigitalOutputDevice(19)
in4 = DigitalOutputDevice(20)
ena = PWMOutputDevice(22, frequency=1000)

def stop_motors():
    in1.off()
    in2.off()
    in3.off()
    in4.off()
    ena.value = 0
    print("Motors STOPPED")

# ---------------- EYE ASPECT RATIO ----------------
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# ---------------- MAIN FUNCTION ----------------
def main():

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # ---------- CONSTANTS ----------
    EYE_AR_THRESH = 0.33
    EYE_AR_CONSEC_FRAMES = 3
    LONG_BLINK_FRAMES = 60
    STOP_BLINKS = 3

    threshold_x = 20
    threshold_y = 20

    COUNTER = 0
    TOTAL = 0
    detection_state = "IDLE"
    locked_direction = ""

    last_nose_x, last_nose_y = -1, -1

    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)

    try:
        while True:
            frame = vs.read()
            if frame is None:
                break

            frame = cv2.flip(frame, 1)
            frame = adjust_gamma(frame, 2.0)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            ear = 0.0
            h, w, _ = frame.shape

            if results.face_landmarks:
                lm = results.face_landmarks.landmark

                left_eye = np.array([
                    (lm[33].x*w, lm[33].y*h),
                    (lm[161].x*w, lm[161].y*h),
                    (lm[159].x*w, lm[159].y*h),
                    (lm[155].x*w, lm[155].y*h),
                    (lm[145].x*w, lm[145].y*h),
                    (lm[144].x*w, lm[144].y*h)
                ])

                right_eye = np.array([
                    (lm[362].x*w, lm[362].y*h),
                    (lm[384].x*w, lm[384].y*h),
                    (lm[386].x*w, lm[386].y*h),
                    (lm[387].x*w, lm[387].y*h),
                    (lm[388].x*w, lm[388].y*h),
                    (lm[380].x*w, lm[380].y*h)
                ])

                ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2

                if ear < EYE_AR_THRESH:
                    COUNTER += 1
                else:
                    if EYE_AR_CONSEC_FRAMES <= COUNTER < LONG_BLINK_FRAMES:
                        TOTAL += 1
                        print("Blink:", TOTAL)

                        if TOTAL == STOP_BLINKS:
                            stop_motors()
                            TOTAL = 0
                            COUNTER = 0
                            detection_state = "IDLE"

                        elif TOTAL == 2:
                            detection_state = "ARMED"
                            last_nose_x = lm[1].x
                            last_nose_y = lm[1].y

                    COUNTER = 0

                if detection_state == "ARMED":
                    dx = (lm[1].x - last_nose_x) * w
                    dy = (lm[1].y - last_nose_y) * h

                    if dx < -threshold_x:
                        print("LEFT")
                        in1.on(); in2.off()
                        in3.off(); in4.off()
                        ena.value = 0.9
                        detection_state = "IDLE"
                        TOTAL = 0

                    elif dx > threshold_x:
                        print("RIGHT")
                        in1.off(); in2.off()
                        in3.on(); in4.off()
                        ena.value = 0.9
                        detection_state = "IDLE"
                        TOTAL = 0

                    elif dy > threshold_y:
                        print("BACK")
                        in1.off(); in2.on()
                        in3.off(); in4.on()
                        ena.value = 0.9
                        detection_state = "IDLE"
                        TOTAL = 0

                    elif dy < -threshold_y:
                        print("FRONT")
                        in1.on(); in2.off()
                        in3.on(); in4.off()
                        ena.value = 0.9
                        detection_state = "IDLE"
                        TOTAL = 0

            cv2.putText(frame, f"Blinks: {TOTAL}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            cv2.imshow("Blink & Head Control", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        stop_motors()
        cv2.destroyAllWindows()
        vs.stop()
        holistic.close()

# ---------------- RUN ----------------
if __name__ == "__main__":
    main()