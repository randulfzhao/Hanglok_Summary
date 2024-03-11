import cv2
import mediapipe as mp
from util.models import myUR5

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
import threading

warnings.filterwarnings('ignore')

def angle(A, B=None, C=None):
    """计算两个三维向量之间的角度，可以接受两种不同的参数格式。"""
    if not B and not C:
        dis1, dis2, dis3 = np.array(A[0]), np.array(A[1]), np.array(A[2])
    else:
        dis1, dis2, dis3 = np.array(A), np.array(B), np.array(C)
    cos_theta = (np.linalg.norm(dis2 - dis3)**2 + np.linalg.norm(dis1 - dis3)**2 - np.linalg.norm(dis1 - dis2)**2) / (2 * np.linalg.norm(dis2 - dis3) * np.linalg.norm(dis1 - dis3))
    return np.arccos(cos_theta)

def draw_hand(image, hand_landmarks):
    mp.solutions.drawing_utils.draw_landmarks(
        image,
        hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style())
def draw_face(image, face_landmarks):
    mp.solutions.drawing_utils.draw_landmarks(
        image,
        face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style())
def draw_pose(image, pose_landmarks):
    mp.solutions.drawing_utils.draw_landmarks(
        image,
        pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_pose_landmarks_style())

def move_robot(arm, angular_velocity, duration):
    global robot_moving
    arm.move_by_v(angular_velocity, duration)
    # time.sleep(duration)
    print(f"Moving at {angular_velocity} has finished")
    robot_moving = False

def position_mapping(joint_angles):
    """
    input: 4 dof list formatting input of angle data of left arm, 
        for shoulder, elbow and wrist
    output: 6 dof list formatting output of angle data for ur5e robot arm, 
        respectively for shoulder_pan_joint, shoulder_lift_joint, elbow_joint, 
        wrist_1_joint, wrist_2_joint and wrist_3_joint
    """

    # joint_angles = [(i-3.14) for i in joint_angles]
    # joint_angles = [i for i in joint_angles]
    joint_angles[0] = joint_angles[0]*(3.14/2-.1)/3.14-3.14/2+.05
    joint_angles[1] -= 3.14/2
    joint_angles[2] = ((joint_angles[2] - 0) * (3.14+.05 - (3.14*3/2-.05)) / (3.14 - 0)) + 3.14*3/2
    joint_angles[3] = (joint_angles[3] - 3.14) * 2
    mapped_joint_angles = [0]+joint_angles+[0]
    return mapped_joint_angles

def angular_mapping(angular_velocity, duration, SPEED_LIMIT):
    # 检查是否超过角速度上限
    max_angular_velocity = max(abs(val) for val in angular_velocity.values())
    # if max_angular_velocity > SPEED_LIMIT:
    scaling_factor = SPEED_LIMIT / max_angular_velocity
    duration /= scaling_factor
    for key in angular_velocity.keys():
        angular_velocity[key] *= scaling_factor        

    mapped_angular_velocity = [0]+[angular_velocity['shoulder'],angular_velocity['elbow'],angular_velocity['wrist'],angular_velocity['finger']] + [0]
    return mapped_angular_velocity, duration

# initialization
_, _, pipeline = rs_initialize()
check_dirs()
old_angles = {'shoulder': 0, 'elbow': 0, 'wrist': 0, 'finger': 0}
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
robot_moving = False
SPEED_LIMIT = 0.2  # 将角速度上限设为 0.2 rad/s

# For webcam input:
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
    arm = myUR5()  # Initializing the robot
    old_time = time.time()
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

        # print(f"The landmarks' condition is {results.pose_landmarks.landmark}")

        try:
            if not robot_moving:
                vis = True
                landmarks = [results.pose_landmarks.landmark[23], # body
                            results.pose_landmarks.landmark[11], # shoulder
                            results.pose_landmarks.landmark[13], # elbow
                            results.left_hand_landmarks.landmark[0], # wrist
                            results.left_hand_landmarks.landmark[5], # mid_hand 1
                            results.left_hand_landmarks.landmark[9], # mid_hand 2
                            results.left_hand_landmarks.landmark[13], # mid_hand 3
                            results.left_hand_landmarks.landmark[17], # mid_hand 4
                            results.left_hand_landmarks.landmark[8], # top_hand 1
                            results.left_hand_landmarks.landmark[12], # top_hand 2
                            results.left_hand_landmarks.landmark[16], # top_hand 3
                            results.left_hand_landmarks.landmark[20]] # top_hand 4
                for landmarki in landmarks[0:3]:
                    if landmarki.visibility < .8:
                        vis = False
                        break
                if not vis:
                    continue

                extracted = [np.array([item.x, item.y, item.z]) for item in landmarks]

                ind_ang = {'shoulder': [0,1,2], 'elbow': [1,2,3], 'wrist': [2,3,4], 'finger': [3,4,5]}
                hand_pos = (extracted[4] + extracted[5] + extracted[6] + extracted[7])/4
                extracted[4] = hand_pos
                del extracted[5:8]
                finger_pos = (extracted[5] + extracted[6] + extracted[7] + extracted[8])/4
                extracted[5] = finger_pos
                del extracted[6:]

                angles = dict()
                for key,value in ind_ang.items():
                    pos = [extracted[i] for i in value]
                    angle_i = angle(pos)
                    angles[key] = angle_i
                
                current_time = time.time()
                delta_time = current_time - old_time

                angular_velocity = {}
                for key in angles.keys():
                    angular_velocity[key] = (angles[key] - old_angles[key]) / delta_time

                # 将机器人设为移动状态并通过多线程设置角速度
                robot_moving = True

                angular_velocity, delta_time = angular_mapping(angular_velocity,delta_time,SPEED_LIMIT)
                
                # move_robot(arm, angular_velocity, delta_time)
                
                thread = threading.Thread(target=move_robot, args=(arm, angular_velocity, delta_time))
                thread.start()

                # 更新旧角度和时间
                old_angles = angles
                old_time = current_time
        except Exception as e:
            # print(e)
            continue


cap.release()