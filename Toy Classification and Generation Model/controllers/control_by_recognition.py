# Standard library imports
import os
import time
import warnings
import ast
import pickle
from datetime import datetime
from collections import deque

# Third-party imports
import cv2
import mediapipe as mp
import pyrealsense2 as rs
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# Local application/library specific imports
from util.models import *
from util.func import *
import threading

def classify_and_predict(df, num_keyframe, loaded_classifier, loaded_models_dict, loaded_encoded_to_label, device, arm):
    df = process_current_df(df, num_keyframe)
    initial_point = df[0]
    initial_point = np.array(initial_point).ravel()

    # Classify the action using the loaded classifier
    data_tensor = torch.tensor(df, dtype=torch.float32).to(device)
    data_tensor = data_tensor.unsqueeze(0)
    with torch.no_grad():
        outputs = loaded_classifier(data_tensor)
        predicted_class = torch.argmax(outputs, dim=1).item()   

    print(f"Predicted class for the action trajectory: {predicted_class}")
    
    # Predict the trajectory for robotic arm movements based on the classified action
    trajectory = TrajectoryModel.generate_trajectory(loaded_models_dict, loaded_encoded_to_label[predicted_class], initial_point, device)

    new_traj = [i[0] for i in trajectory]
    for point in new_traj:
        arm.move_to_point(point)

"""
基于规则的方法：
如果你的动作有明显的开始和结束特征，你可以定义一些规则来划分动作。
例如，你可以定义当手部的某个关键点达到某个位置时，认为动作开始；
当手部的某个关键点达到另一个位置时，认为动作结束。
"""

# Set the environment variable to prevent certain errors when using specific libraries (often related to TensorFlow)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Filter out any warnings for cleaner console output
warnings.filterwarnings('ignore')

# Configuration Parameters:
num_keyframe = 221
INPUT_DIM = 21 * 3  # Assuming 21 keypoints with 3D coordinates
HIDDEN_DIM = 128
NUM_LAYERS = 2
NUM_EPOCHS = 50
BATCH_SIZE = 64
OUTPUT_DIM = 3
EARLY_STOP_PATIENCE = 10
LEARNING_RATE = 0.001

# Model Path:
# Ensure to replace the path with the location where your trained model is stored.
classify_model_path = "saved_models/model_checkpoint.pth"
loaded_classifier = ActionClassifier.load(classify_model_path, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS)

# Load label dictionaries for the classifier
loaded_label_to_encoded, loaded_encoded_to_label = load_label_dicts()

# Setup for GPU (if available) or default to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trajectory models for different classes
loaded_models_dict = TrajectoryModel.load(loaded_label_to_encoded, device)

# Initialization of MediaPipe for hand tracking and RealSense for depth sensing
mp_drawing, mp_hands, pipeline = rs_initialize()

# Check and potentially create required directories for storing files
check_dirs()

# Video file setup, DataFrame initialization, and control flags
out = None
df_list = []
recording = False
start_detected = False
end_detected = False
end_gesture_detected_time = None

# Initialize robotic arm interface
arm = myUR5()

# Begin hand tracking
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while True:
        # Acquire frames from the RealSense camera
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert frames to image format suitable for processing
        image = np.asanyarray(color_frame.get_data())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image through the MediaPipe hand tracking model
        results = hands.process(image)

        # Convert image back to BGR for visualization
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # If hand landmarks are detected, process them
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the hand landmarks on the image
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Detect start gesture for recording
                if is_start_gesture(hand_landmarks, mp_hands) and not recording and not start_detected:
                    print('Start gesture detected.')
                    filename = get_current_time()
                    out = cv2.VideoWriter(f'vid/{filename}.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))
                    df_list = []
                    recording = True
                    start_detected = True
                    end_detected = False

                # Detect end gesture to stop recording
                elif is_end_gesture(hand_landmarks, mp_hands) and start_detected and recording and not end_detected:
                    print('End gesture detected.')
                    end_detected = True
                    end_gesture_detected_time = time.time()

                # Stop recording 0.5 seconds after the end gesture is detected
                if end_gesture_detected_time and time.time() - end_gesture_detected_time > 0.5:
                    if len(df_list) > 15:  # Assuming 30fps, 0.5 seconds would result in 15 frames
                        # Save the video and process the DataFrame
                        out.release()
                        df = pd.DataFrame(df_list)
                        df.to_csv(f'excel/{filename}.csv', index=False)
                        thread = threading.Thread(target=classify_and_predict, 
                                                    args=(df, num_keyframe, loaded_classifier, loaded_models_dict, 
                                                    loaded_encoded_to_label, device, arm))
                        thread.start()


                    # Reset recording flags
                    recording = False
                    start_detected = False
                    end_detected = False
                    end_gesture_detected_time = None

                # If in recording mode, save the hand landmarks for later processing
                if recording:
                    hand_data = []
                    for lm in hand_landmarks.landmark:
                        hand_data.append([lm.x, lm.y, lm.z])
                    df_list.append(hand_data)

        # Write the frame with hand landmarks to the video file
        if recording:
            out.write(image)

        # Show the processed image with hand landmarks
        cv2.imshow('MediaPipe Hands', image)

        # If 'ESC' key is pressed, break out of the loop
        if cv2.waitKey(5) & 0xFF == 27:
            break

# If still recording at the end, save the data
if recording and len(df_list) > 15:
    out.release()
    df = pd.DataFrame(df_list)
    df.to_csv(f'excel/{filename}.csv', index=False)

# Release resources and close windows
pipeline.stop()
cv2.destroyAllWindows()
