# Importing necessary libraries
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from pyrealsense2 import pyrealsense2 as rs
from controller import Robot
from ikpy.chain import Chain
from util.models import myUR5

original = list()
after = list()

# MediaPipe related settings
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)  # Initializing the hand detection model

# Setting up RealSense camera
pipe = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipe.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# Getting camera intrinsics
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

# Function to rescale values for transformation
def rescale(value, old_min, old_max, new_min=-1.1, new_max=1.1):
    return ((value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min

if __name__ == "__main__":
    arm = myUR5()  # Initializing the robot

    try:
        while True:  # When RGB-D camera is active
            frames = pipe.wait_for_frames()  # Read frames
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # Convert the BGR image to RGB before processing
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            
            results = hands.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks:  # If hand landmarks are found
                for hand_landmarks in results.multi_hand_landmarks:
                    for id, lm in enumerate(hand_landmarks.landmark):
                        if id == mp_hands.HandLandmark.WRIST.value:  # If it's the wrist landmark
                            wrist_x = int(lm.x * color_image.shape[1])
                            wrist_y = int(lm.y * color_image.shape[0])
                            wrist_x = max(0, min(719, wrist_x))  # Assuming the width is also 720, adjust according to actual settings
                            wrist_y = max(0, min(719, wrist_y))
                            wrist_z = depth_image[wrist_y, wrist_x].astype(float)
                            if wrist_z == 0:
                                continue

                            original_position = [wrist_x, wrist_y, wrist_z]
                            original.append(original_position)

                            # Convert pixel coordinates to real-world coordinates, adjustments may be needed later
                            x_rescaled = rescale(wrist_x, 109, 588)
                            y_rescaled = rescale(wrist_y, 36, 476)
                            z_rescaled = rescale(wrist_z, 500, 1200)

                            # Transform coordinate systems, and may need to scale proportionally
                            scale_factor = 1  # Adjust based on actual needs
                            target_position = [x_rescaled, z_rescaled, -y_rescaled]
                            after.append(target_position)
                            joint_positions = arm.inverse_kinematics(target_position)
                            arm.set_joint_positions(joint_positions)

            # Drawing the hand landmarks on the image
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Display the images
            cv2.imshow('MediaPipe Hands', color_image)
            cv2.imshow('Depth Image', depth_colormap)
            if cv2.waitKey(5) & 0xFF == 27:  # Exit when 'ESC' is pressed
                outputs = pd.DataFrame({'original': original, 'after': after})
                outputs.to_csv("positions.csv", index=False)
                break

    finally:
        pipe.stop()  # Shut down the RGB-D camera
        cv2.destroyAllWindows()  # Close all open windows
