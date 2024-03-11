import cv2
import mediapipe as mp

import cv2
import mediapipe as mp
import pyrealsense2 as rs
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
import warnings
from util.func import *
warnings.filterwarnings('ignore')


_,_,pipeline = rs_initialize()
check_dirs()


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
def draw_hand(image, hand_landmarks):
    mp_drawing.draw_landmarks(
        image,
        hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style())
def draw_face(image, face_landmarks):
    mp_drawing.draw_landmarks(
        image,
        face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style())

def draw_pose(image, pose_landmarks):
    mp_drawing.draw_landmarks(
        image,
        pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_pose_landmarks_style())

# For webcam input:
cap = cv2.VideoCapture(0)

# Initialize video file, DataFrame and related flags
out = None
df_list = []
recording = False
start_detected = False
end_detected = False
end_gesture_detected_time = None

with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        # Draw landmark annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        draw_face(image, results.face_landmarks)
        draw_pose(image, results.pose_landmarks)
        draw_hand(image, results.right_hand_landmarks)
        draw_hand(image, results.left_hand_landmarks)
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break


        # Detect the starting gesture
        if is_start_gesture(results.left_hand_landmarks, mp_holistic.HandLandmark) \
            or is_start_gesture(results.right_hand_landmarks, mp_holistic.HandLandmark):  # You need to implement is_start_gesture for poses
            if not recording and not start_detected:
                print('Start gesture detected.')
                filename = get_current_time()
                out = cv2.VideoWriter(f'vid/{filename}.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))
                df_list = []
                recording = True
                start_detected = True
                end_detected = False
        # Detect the ending gesture
        elif is_end_gesture(results.left_hand_landmarks, mp_holistic.HandLandmark) \
            or is_end_gesture(results.right_hand_landmarks, mp_holistic.HandLandmark):  # You need to implement is_end_gesture for poses
            if start_detected and recording and not end_detected:
                print('End gesture detected.')
                end_detected = True
                end_gesture_detected_time = time.time()

        # Stop recording 0.5 seconds after the end gesture is detected
        if end_gesture_detected_time and time.time() - end_gesture_detected_time > 0.5:
            if len(df_list) > 15:
                out.release()
                df = pd.DataFrame(df_list)
                df.to_csv(f'excel/{filename}.csv', index=False)
            recording = False
            start_detected = False
            end_detected = False
            end_gesture_detected_time = None


        # If recording is in progress, save the hand landmark data
        if recording:
            ind = [11,13,15,17,19,21]
            extracted = [[results.pose_landmarks.landmark[i].x, results.pose_landmarks.landmark[i].y, results.pose_landmarks.landmark[i].z] for i in ind]
            df_list.append(extracted)


# If recording is in progress when the loop ends, save the data
if recording:
    if len(df_list) > 15:
        out.release()
        df = pd.DataFrame(df_list)
        df.to_csv(f'excel/{filename}.csv', index=False)

cap.release()
