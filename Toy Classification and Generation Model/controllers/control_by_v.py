import cv2
import mediapipe as mp
import pyrealsense2 as rs
import numpy as np
from collections import deque
from datetime import datetime
from util.models import *
from util.func import *

def calculate_velocity_and_direction(previous_position, current_position):
    """
    Calculate the velocity and direction based on the difference between two 3D positions.
    
    Parameters:
    - previous_position (list): The 3D position at the earlier time.
    - current_position (list): The 3D position at the later time.
    
    Returns:
    - float: The calculated speed.
    - str: The dominant direction of motion.
    """
    
    # Calculating displacements in the x, y, z dimensions
    delta_x = current_position[0] - previous_position[0]
    delta_y = current_position[1] - previous_position[1]
    delta_z = current_position[2] - previous_position[2]

    # Calculate the magnitude of the speed
    speed = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)

    # Determine direction based on displacements
    directions = {
        '左': abs(delta_x) * (-1 if delta_x < 0 else 1),
        '右': abs(delta_x),
        '上': abs(delta_y),
        '下': abs(delta_y) * (-1 if delta_y < 0 else 1),
        '前': abs(delta_z),
        '后': abs(delta_z) * (-1 if delta_z < 0 else 1),
    }

    # Retrieve the direction with the highest magnitude
    direction = max(directions, key=directions.get)

    return speed, direction

# Defining dictionary for direction vectors
direction_vector_dict = {
    '左': [-1, 0, 0],
    '右': [1, 0, 0],
    '上': [0, 1, 0],
    '下': [0, -1, 0],
    '前': [0, 0, -1],
    '后': [0, 0, 1]
}

# Define the speed threshold for further actions
SPEED_THRESHOLD = 0.05  

# Initialize MediaPipe and RealSense
mp_drawing, mp_hands, pipeline = rs_initialize()

# Sliding window to store recent wrist positions
window = deque(maxlen=15)  # Assuming 30 fps, so 0.5s has 15 frames

# Counter to keep track of processed frames
frame_counter = 0
# Interval for velocity calculations
calculation_interval = 15  # Since we have 30fps, this means we calculate once every 0.5 seconds
arm = myUR5()
arm.set_pos([.5,.5,.5,.5,.5,.2])

# Start processing each frame for hand detection
with mp_hands.Hands(min_detection_confidence=.7, min_tracking_confidence=.7) as hands:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        image = np.asanyarray(color_frame.get_data())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Check if any hand landmarks are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get wrist position in 3D space
                wrist_position = [hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x,
                                  hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y,
                                  hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z]

                window.append(wrist_position)

                frame_counter += 1
                if frame_counter == calculation_interval:
                    speed, direction = calculate_velocity_and_direction(window[0], wrist_position)
                    direction_vec = direction_vector_dict[direction]

                    # If the speed exceeds the threshold, then actuate robot arm
                    if speed > SPEED_THRESHOLD:
                        print(f'Direction is {direction_vec}, and speed is {speed}')
                        velocity = np.array(direction_vec)*speed*5
                        arm.move(velocity, duration=.5)

                    frame_counter = 0

        # Display the processed frame
        cv2.imshow('MediaPipe Hands', image)
        
        # Break the loop if the ESC key is pressed
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Stop the pipeline and close windows after processing
pipeline.stop()
cv2.destroyAllWindows()
