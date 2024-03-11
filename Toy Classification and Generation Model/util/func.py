# ----------------- Standard Library -----------------
import os
import ast
import time
import pickle
import warnings
from datetime import datetime
from collections import deque

# ----------------- External Libraries -----------------
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import pyrealsense2 as rs
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# ----------------- Local Modules -----------------
from util.models import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ---------------------------------- Data Preprocessing ----------------------------------
"""
data format:
* $data$: dataset of actions, 15 actions in total
* $data[i]$: $i^{th}$ action with 230 keyframes
* $data[i][j]$: $j^{th}$ keyframe of $i^{th}$ action with 21 keypoints
* $data[i][j][k]$: 3D coordinate information of the $k^{th}$ keypoint for the $j^{th}$ keyframe of $i^{th}$ action
"""

# ---- Utility Functions ----

def extract_action_label(filename, actions=['drinkWater', 'reachOut', 'getPhone']):
    """Extract action label from the filename."""
    for action in actions:
        if action in filename:
            return action
    return None

def parse_string_to_list(entry):
    """Convert a string representation of a list to an actual list."""
    return np.array(ast.literal_eval(entry))

def compute_difference(df):
    """Convert a string representation of a list to an actual list."""
    try:
        df = df[0].apply(parse_string_to_list)
        diff = df.diff().dropna()
    except:
        diff = df.diff().dropna()
    return diff

def magnitude_of_difference(diff):
    """Calculate the magnitude (norm) of the difference for each row."""
    return diff.apply(lambda x: np.linalg.norm(x))

def significant_magnitude(magnitude, threshold_value):
    """Mark entries with magnitude greater than the threshold."""
    return magnitude > threshold_value

def get_keyframes_based_on_difference(df, target_length):
    """Select keyframes based on significant differences in data."""
    diff = compute_difference(df)
    magnitude = magnitude_of_difference(diff)
    threshold = 1 / target_length
    significant = significant_magnitude(magnitude, threshold)
    return df.iloc[significant.nlargest(target_length).index]

# ---- Loading Data Functions----

def load_data_from_directory(directory):
    """ 
    Load CSV files from a given directory.
    
    :param directory: Path of the directory containing CSV files.
    :return: List of file paths and the average length of all files.
    """
    
    # Create a list of complete file paths for every CSV file found in the directory.
    file_paths = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.csv')]
    
    # Get the lengths of each CSV file.
    lengths = [len(pd.read_csv(file, header=None)) for file in file_paths]
    
    # Calculate the average length across all files.
    average_length = int(np.mean(lengths))
    
    return file_paths, average_length

def preprocess_body(file, target_length):
    """ 
    Process the content of a single file. This is a common function used for both training and test data.
    
    :param file: Path of the file to be processed.
    :param target_length: The length that the dataset should conform to.
    :return: Preprocessed data and the corresponding label.
    """
    
    # Read the file content into a DataFrame.
    df = pd.read_csv(file, header=None)
    current_length = len(df)

    # If the data length is more than the target, process it to fit.
    if current_length > target_length:
        df = get_keyframes_based_on_difference(df, target_length)
        
    # Extract labels for the dataset.
    label = extract_action_label(file)

    # Convert the DataFrame to a list of lists.
    df_list = df.values.tolist()
    new_df_list = []  # Will store the processed rows

    # Filter out rows not suitable for the DataFrame.
    for j in range(len(df_list)):
        delete_row = False  # Flag to decide whether to keep the row

        for k in range(len(df_list[j])):
            value = df_list[j][k]
            if isinstance(value, str) and value.isdigit():  # Check if the value is a digit represented as a string.
                delete_row = True
                break  # Exit the inner loop if an unsuitable value is found
            else:
                df_list[j][k] = ast.literal_eval(value)  # Convert strings to corresponding data structures

        if not delete_row:  # If the row is still valid, add it to the new list.
            new_df_list.append(df_list[j])

    # Pad the data if its length is shorter than the target.
    while len(new_df_list) < target_length:
        new_df_list.append(new_df_list[-1])

    return new_df_list, label

def preprocess_train(directory):
    """ 
    Load and preprocess data for training.
    
    :param directory: Path of the directory containing the training data.
    :return: Preprocessed training data, corresponding labels, and average data length.
    """
    
    # Load files and get the average length.
    file_paths, average_length = load_data_from_directory(directory)
    
    processed_data, processed_labels = [], []
    for file in file_paths:
        data, label = preprocess_body(file, average_length)
        processed_data.append(data)
        processed_labels.append(label)
        
    return processed_data, processed_labels, average_length

def preprocess_test(directory, keyframe):
    """ 
    Load and preprocess data for testing.
    
    :param directory: Path of the directory containing the test data.
    :param keyframe: Length that the test data should conform to.
    :return: Preprocessed test data and corresponding labels.
    """
    
    # Load test files.
    file_paths, _ = load_data_from_directory(directory)
    
    processed_data, processed_labels = [], []
    for file in file_paths:
        data, label = preprocess_body(file, keyframe)
        processed_data.append(data)
        processed_labels.append(label)
        
    return processed_data, processed_labels

def get_keyframes_current(df, keyframe):
    """
    This function extracts the significant frames (keyframes) from a given dataframe.
    
    :param df: DataFrame containing the data.
    :param keyframe: The desired number of keyframes to extract.
    :return: DataFrame with rows corresponding to the most significant changes.
    """
    
    # Calculate the difference between consecutive rows of the DataFrame.
    # This gives us how much each element has changed from the previous row.
    diff = [[i - j for i, j in zip(curr, prev)] for curr, prev in zip(df[0][1:], df[0][:-1])]
    
    df_diff = pd.DataFrame()
    
    # The 'diff' list will have one less item than the original DataFrame since it's a difference between consecutive items.
    # Therefore, we add an np.nan at the beginning to make their lengths equal.
    df_diff['diff'] = [np.nan] + diff 
    
    # Compute the magnitude (Euclidean norm) of the difference for each row.
    # If the item is not a list (like np.nan), then set it to np.nan.
    magnitude = df_diff['diff'].apply(lambda x: np.linalg.norm(np.array(x)) if isinstance(x, list) else np.nan)

    # Define a threshold for significant changes. Changes with magnitude greater than this threshold will be considered significant.
    threshold = 1 / keyframe
    significant = magnitude > threshold
    
    # Return rows from the original DataFrame that correspond to the largest significant changes.
    return df.iloc[significant.nlargest(keyframe).index]

def process_current_df(df, keyframe):
    """
    This function processes a DataFrame and ensures it has the desired number of keyframes.
    
    :param df: DataFrame to be processed.
    :param keyframe: Desired number of keyframes.
    :return: List of values from the processed DataFrame.
    """
    
    current_length = len(df)
    
    # If the DataFrame has more rows than the desired number of keyframes, 
    # extract the keyframes using the get_keyframes_current function.
    if current_length > keyframe:
        df = get_keyframes_current(df, keyframe) 

    # Convert the DataFrame to a list.
    df_list = df.values.tolist()
    
    # If the number of keyframes is less than the desired number, 
    # duplicate the last keyframe until the desired number is reached.
    while len(df_list) < keyframe:
        df_list.append(df_list[-1])
    
    return df_list

# ---- Time-Related Functions ----

def get_current_time():
    """Return the current time as a formatted string."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')

# ---- Label Encoding Functions ----

def save_label_dicts(label_to_encoded, encoded_to_label, save_dir="saved_dicts"):
    """Save label encoding dictionaries to disk."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, 'label_to_encoded.pkl'), 'wb') as f:
        pickle.dump(label_to_encoded, f)
    with open(os.path.join(save_dir, 'encoded_to_label.pkl'), 'wb') as f:
        pickle.dump(encoded_to_label, f)

def load_label_dicts(save_dir="saved_dicts"):
    """Load label encoding dictionaries from disk."""
    with open(os.path.join(save_dir, 'label_to_encoded.pkl'), 'rb') as f:
        label_to_encoded = pickle.load(f)
    with open(os.path.join(save_dir, 'encoded_to_label.pkl'), 'rb') as f:
        encoded_to_label = pickle.load(f)
    return label_to_encoded, encoded_to_label


# ---------------------------------- Data Augumentation ----------------------------------

def add_noise(points, sigma=0.01):
    """Add noise to the data points.
    
    Args:
    - points: Original list or array of data points.
    - sigma: Standard deviation of the noise.
    
    Returns:
    - Data points with added noise.
    """
    points_np = np.array(points)
    noise = np.random.normal(0, sigma, points_np.shape)
    return points_np + noise

def scale(points, scale_factor=None):
    """Scale the data points.
    
    Args:
    - points: Original list or array of data points.
    - scale_factor: Scaling factor. If None, a random factor is chosen.
    
    Returns:
    - Scaled data points.
    """
    points_np = np.array(points)
    scale_factor = scale_factor or np.random.uniform(0.9, 1.1)
    return points_np * scale_factor

def rotate(points, degree_range=10):
    """Rotate three-dimensional data points.
    
    Args:
    - points: Original list or array of data points.
    - degree_range: Rotation degree range. A degree is randomly chosen from this range.
    
    Returns:
    - Rotated data points. If data isn't three-dimensional, returns the original data.
    """
    points_np = np.array(points)
    
    if points_np.shape[-1] != 3:
        return points_np
    
    angles = np.radians(np.random.uniform(-degree_range, degree_range, 3))
    rotation_matrices = [
        np.array([[1, 0, 0], [0, np.cos(angles[0]), -np.sin(angles[0])], [0, np.sin(angles[0]), np.cos(angles[0])]]),
        np.array([[np.cos(angles[1]), 0, np.sin(angles[1])], [0, 1, 0], [-np.sin(angles[1]), 0, np.cos(angles[1])]]),
        np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0], [np.sin(angles[2]), np.cos(angles[2]), 0], [0, 0, 1]])
    ]
    
    rotation_matrix = np.linalg.multi_dot(rotation_matrices)
    return np.dot(points_np, rotation_matrix.T)

def translate(points, max_translation=0.1):
    """Translate three-dimensional data points.
    
    Args:
    - points: Original list or array of data points.
    - max_translation: Maximum translation distance.
    
    Returns:
    - Translated data points. If data isn't three-dimensional, returns the original data.
    """
    points_np = np.array(points)
    
    if points_np.shape[-1] != 3:
        return points_np
    
    translation = np.random.uniform(-max_translation, max_translation, 3)
    return points_np + translation

def augment_single_action(action, times=5):
    """Augment the data of a single action multiple times.
    
    Args:
    - action: Original list or array of action data.
    - times: Number of augmentations.
    
    Returns:
    - List of augmented action data.
    """
    augmented_actions = [action]
    
    for _ in range(times):
        augmented_action = action.copy()
        augmentation_functions = [add_noise, scale, rotate, translate]
        
        for func in augmentation_functions:
            augmented_action = [func(keyframe) for keyframe in augmented_action]
        
        augmented_actions.append(augmented_action)
    
    return augmented_actions

def augment_data_and_labels(data, labels, times=5):
    """Augment an entire dataset and its labels.
    
    Args:
    - data: Original list or array of action data.
    - labels: Corresponding list of labels.
    - times: Number of augmentations per action.
    
    Returns:
    - Lists of augmented data and labels.
    """
    augmented_data = []
    augmented_labels = []

    for action, label in zip(data, labels):
        new_actions = augment_single_action(action, times)
        augmented_data.extend(new_actions)
        augmented_labels.extend([label] * len(new_actions))

    return augmented_data, augmented_labels

# ---------------------------------- File Operation ----------------------------------

def sync_excel_to_vid(dir_video,dir_excel):
    """Sync Filename from directory of video to that of excel"""
    # Retrieve the list of all filenames in the directory of video documents and that of excel documents.
    vid_files = os.listdir(dir_video)  # List of files in the dir_video directory.
    excel_files = os.listdir(dir_excel)  # List of files in the dir_excel directory.
    
    # Loop through all the files in the directory of excel document.
    for excel_file in excel_files:
        # Extract the filename without the file extension.
        base_name = os.path.splitext(excel_file)[0]
        
        # Find all video files in the directory of video document that match the current Excel file's base name.
        matched_video_files = [v for v in vid_files if base_name in v]
        
        # If a matching video file for the current Excel file is found in the directory of video documents.
        if matched_video_files:
            # Ensure only one matching video file is found.
            if len(matched_video_files) == 1:
                matched_video_file = matched_video_files[0]
                # Create a new filename using the matched video's name but with a .csv extension.
                new_excel_name = os.path.splitext(matched_video_file)[0] + '.csv'
                
                # Rename the current Excel file using the new filename.
                os.rename(os.path.join(dir_excel, excel_file), os.path.join(dir_excel, new_excel_name))
            else:
                # If multiple matching video files are found, print an informative message.
                print(f"Multiple matching files for '{base_name}' found in directory of video documents: {dir_video}. Uncertain which one to use.")
        # If no matching video file for the current Excel file is found in the dir_video directory.
        else:
            # Delete the Excel file and print an informative message.
            os.remove(os.path.join(dir_excel, excel_file))
            print(f"File '{excel_file}' has been removed from the directory of excel documents: {dir_excel}.")

def count_files_in_directory(dir_path='vid', suffixes=['drinkWater', 'reachOut', 'getPhone']):
    """
    Count the number of files in a directory based on predefined suffixes.

    Args:
    - dir_path (str): Path to the directory to be scanned. Default is 'vid'.
    - suffixes (list of str): Categories/Types of file endings to search for. Default is ['drinkWater', 'reachOut', 'getPhone'].

    Note:
    - This function assumes that files only match one of the given suffixes.
    - The function also prints the count for each motion type and the total number of files.
    """

    # Initialize a dictionary to hold the count of files for each suffix.
    counts = {suffix: 0 for suffix in suffixes}
    
    # Initialize a variable to count the total number of files encountered.
    total_files = 0

    # Iterate over each file in the specified directory.
    for file_name in os.listdir(dir_path):
        # Check each file against the predefined suffixes.
        for suffix in suffixes:
            # If the current suffix is found in the file name, increment its count and the total files count.
            if suffix in file_name:
                counts[suffix] += 1
                total_files += 1
                # Once a match is found, break out of the inner loop to avoid double-counting a file for multiple suffixes.
                break  

    # Print the results: count for each type of motion and the total count.
    for suffix, count in counts.items():
        print(f"Number of samples for motion '{suffix}' is {count}")
    print(f"Total files: {total_files}")

def check_dirs(video_dir='vid', excel_path='excel'):
    """
    Checks if the specified directories exist. If not, it creates them.
    
    Parameters:
    - video_dir (str): The path for the video directory. Default is 'vid'.
    - excel_path (str): The path for the Excel directory. Default is 'excel'.
    """
    
    # Check for the existence of video directory, create if doesn't exist
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    
    # Check for the existence of excel directory, create if doesn't exist
    if not os.path.exists(excel_path):
        os.makedirs(excel_path)


# ---------------------------------- Real Sense Camera's Operations ----------------------------------

def rs_initialize(depth_config = [640,480], color_config = [640,480]):
    """
    Initializes the mediapipe drawing utilities, hand solutions, and RealSense pipeline with the given configurations.
    
    Parameters:
    - depth_config (list): Contains the width and height configuration for the depth stream. Default is [640, 480].
    - color_config (list): Contains the width and height configuration for the color stream. Default is [640, 480].
    
    Returns:
    - tuple: Contains the drawing utility, hand solution, and RealSense pipeline objects.
    """
    
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    
    # Initialize the RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Configure the streams for depth and color
    config.enable_stream(rs.stream.depth, depth_config[0], depth_config[1], rs.format.z16, 30)
    config.enable_stream(rs.stream.color, color_config[0], color_config[1], rs.format.bgr8, 30)
    pipeline.start(config)

    return mp_drawing, mp_hands, pipeline

# def is_start_gesture(hand_landmarks, source, threshold=.1):
#     """
#     Checks if the hand gesture corresponds to the "start" gesture.
    
#     Parameters:
#     - hand_landmarks (object): Contains the landmark data for the hand.
#     - threshold (float): The distance threshold to determine the start gesture.
    
#     Returns:
#     - bool: True if the distance between thumb and index finger is less than the threshold, False otherwise.
#     """
    
#     try:
#         if hand_landmarks.landmark[source.THUMB_TIP].visibility>.5 and\
#             hand_landmarks.landmark[source.INDEX_FINGER_TIP].visibility>.5 :
#             # Extracting the x, y, z coordinates for the thumb tip
#             thumb_tip = [hand_landmarks.landmark[source.THUMB_TIP].x,
#                         hand_landmarks.landmark[source.THUMB_TIP].y,
#                         hand_landmarks.landmark[source.THUMB_TIP].z]
            
#             # Extracting the x, y, z coordinates for the index finger tip
#             index_finger_tip = [hand_landmarks.landmark[source.INDEX_FINGER_TIP].x,
#                                 hand_landmarks.landmark[source.INDEX_FINGER_TIP].y,
#                                 hand_landmarks.landmark[source.INDEX_FINGER_TIP].z]
            
#             # Calculating the euclidean distance between the thumb tip and index finger tip
#             distance = np.sqrt(np.sum(np.square(np.subtract(thumb_tip, index_finger_tip))))

#             # Return True if distance is less than threshold, indicating the start gesture
#             return distance < threshold
#         else:
#             return False
#     except:
#         return False

# def is_end_gesture(hand_landmarks, source, threshold=.1):
#     """
#     Checks if the hand gesture corresponds to the "end" gesture.
    
#     Parameters:
#     - hand_landmarks (object): Contains the landmark data for the hand.
#     - threshold (float): The distance threshold to determine the end gesture.
    
#     Returns:
#     - bool: True if the distance between thumb and pinky finger is less than the threshold, False otherwise.
#     """

#     try:
#         if hand_landmarks.landmark[source.THUMB_TIP].visibility>.5 and\
#             hand_landmarks.landmark[source.PINKY_TIP].visibility>.5 :
#             # Extracting the x, y, z coordinates for the thumb tip
#             thumb_tip = [hand_landmarks.landmark[source.THUMB_TIP].x,
#                         hand_landmarks.landmark[source.THUMB_TIP].y,
#                         hand_landmarks.landmark[source.THUMB_TIP].z]
            
#             # Extracting the x, y, z coordinates for the pinky finger tip
#             pinky_finger_tip = [hand_landmarks.landmark[source.PINKY_TIP].x,
#                                 hand_landmarks.landmark[source.PINKY_TIP].y,
#                                 hand_landmarks.landmark[source.PINKY_TIP].z]
            
#             # Calculating the euclidean distance between the thumb tip and pinky finger tip
#             distance = np.sqrt(np.sum(np.square(np.subtract(thumb_tip, pinky_finger_tip))))

#             # Return True if distance is less than threshold, indicating the end gesture
#             return distance < threshold
#         else:
#             return False
#     except:
#         return False

def is_start_gesture(hand_landmarks, source, threshold=.05):
    """
    Checks if the hand gesture corresponds to the "start" gesture.
    
    Parameters:
    - hand_landmarks (object): Contains the landmark data for the hand.
    - threshold (float): The distance threshold to determine the start gesture.
    
    Returns:
    - bool: True if the distance between thumb and index finger is less than the threshold, False otherwise.
    """
    
    try:
        # Extracting the x, y, z coordinates for the thumb tip
        thumb_tip = [hand_landmarks.landmark[source.THUMB_TIP].x,
                    hand_landmarks.landmark[source.THUMB_TIP].y,
                    hand_landmarks.landmark[source.THUMB_TIP].z]
        
        # Extracting the x, y, z coordinates for the index finger tip
        index_finger_tip = [hand_landmarks.landmark[source.INDEX_FINGER_TIP].x,
                            hand_landmarks.landmark[source.INDEX_FINGER_TIP].y,
                            hand_landmarks.landmark[source.INDEX_FINGER_TIP].z]
        
        # Calculating the euclidean distance between the thumb tip and index finger tip
        distance = np.sqrt(np.sum(np.square(np.subtract(thumb_tip, index_finger_tip))))

        # Return True if distance is less than threshold, indicating the start gesture
        return distance < threshold
    except:
        return False

def is_end_gesture(hand_landmarks, source, threshold=.05):
    """
    Checks if the hand gesture corresponds to the "end" gesture.
    
    Parameters:
    - hand_landmarks (object): Contains the landmark data for the hand.
    - threshold (float): The distance threshold to determine the end gesture.
    
    Returns:
    - bool: True if the distance between thumb and pinky finger is less than the threshold, False otherwise.
    """

    try:
        # Extracting the x, y, z coordinates for the thumb tip
        thumb_tip = [hand_landmarks.landmark[source.THUMB_TIP].x,
                    hand_landmarks.landmark[source.THUMB_TIP].y,
                    hand_landmarks.landmark[source.THUMB_TIP].z]
        
        # Extracting the x, y, z coordinates for the pinky finger tip
        pinky_finger_tip = [hand_landmarks.landmark[source.PINKY_TIP].x,
                            hand_landmarks.landmark[source.PINKY_TIP].y,
                            hand_landmarks.landmark[source.PINKY_TIP].z]
        
        # Calculating the euclidean distance between the thumb tip and pinky finger tip
        distance = np.sqrt(np.sum(np.square(np.subtract(thumb_tip, pinky_finger_tip))))

        # Return True if distance is less than threshold, indicating the end gesture
        return distance < threshold
    except:
        return False

# def is_start_gesture(pose_landmarks, threshold=0.2):
#     """
#     Checks if the left arm posture corresponds to the "start" gesture (straight arm).
    
#     Parameters:
#     - pose_landmarks (object): Contains the landmark data for the pose.
#     - threshold (float): The ratio threshold to determine the start gesture.
    
#     Returns:
#     - bool: True if the left wrist is close to right shoulder, False otherwise.
#     """
#     if not pose_landmarks or not hasattr(pose_landmarks, "landmark"):
#         return False
    
#     visibility1 = pose_landmarks.landmark[15].visibility>.8
#     visibility2 = pose_landmarks.landmark[12].visibility>.8

#     # Extracting the coordinates for the left wrist, elbow, and shoulder
#     left_wrist = [pose_landmarks.landmark[15].x,
#              pose_landmarks.landmark[15].y,
#              pose_landmarks.landmark[15].z]
#     right_shoulder = [pose_landmarks.landmark[12].x,
#                 pose_landmarks.landmark[12].y,
#                 pose_landmarks.landmark[12].z]

#     dist = np.sqrt(np.sum(np.square(np.subtract(left_wrist, right_shoulder))))

#     # If the two distances are roughly the same (ratio close to 1), the arm is straight
#     return (dist < threshold) and visibility1 and visibility2 

# def is_end_gesture(pose_landmarks, threshold=0.2):
#     """
#     Checks if the left arm posture corresponds to the "end" gesture (bent arm).
    
#     Parameters:
#     - pose_landmarks (object): Contains the landmark data for the pose.
#     - threshold (float): The ratio threshold to determine the end gesture.
    
#     Returns:
#     - bool: True if the right wrist is close to left shoulder, False otherwise.
#     """
#     if not pose_landmarks or not hasattr(pose_landmarks, "landmark"):
#         return False
#     visibility1 = pose_landmarks.landmark[16].visibility>.8
#     visibility2 = pose_landmarks.landmark[11].visibility>.8
#     # Using the same logic as in the is_start_gesture function
#     right_wrist = [pose_landmarks.landmark[16].x,
#              pose_landmarks.landmark[16].y,
#              pose_landmarks.landmark[16].z]
#     left_shoulder = [pose_landmarks.landmark[11].x,
#                 pose_landmarks.landmark[11].y,
#                 pose_landmarks.landmark[11].z]

#     dist = np.sqrt(np.sum(np.square(np.subtract(right_wrist, left_shoulder))))

#     # If the two distances are roughly the same (ratio close to 1), the arm is straight
#     return dist < threshold and visibility1 and visibility2 
