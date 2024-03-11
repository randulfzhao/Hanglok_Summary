# original models
import os
import ast
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import cv2
import numpy as np
import mediapipe as mp
from pyrealsense2 import pyrealsense2 as rs
from controller import Robot
from ikpy.chain import Chain


# Model for action classification
class ActionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        """Initialize the ActionClassifier model.
        
        Args:
        - input_dim (int): Dimension of the input data.
        - hidden_dim (int): Dimension of the hidden layers.
        - output_dim (int): Dimension of the output (number of classes).
        - num_layers (int): Number of LSTM layers.
        """
        super(ActionClassifier, self).__init__()  # Initialize parent class
        
        self.hidden_dim = hidden_dim  # Set the hidden dimension
        # Define the LSTM layer with the given parameters
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        # Define the fully connected layer that maps LSTM outputs to desired output_dim
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """Forward propagation of the neural network.
        
        Args:
        - x (tensor): The input data tensor.
        
        Returns:
        - tensor: The output tensor after processing.
        """
        # Reshape the input tensor so it's suitable for LSTM layer
        x = x.view(x.size(0), x.size(1), -1)
        # Pass the reshaped input data through the LSTM layer
        out, _ = self.lstm(x)
        # Pass the last output of LSTM (out[:, -1, :]) through the fully connected layer
        out = self.fc(out[:, -1, :])
        return out
    
    @staticmethod
    def load(model_path, input_dim, hidden_dim, output_dim, num_layers, 
                        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """Load a classifier model from the given path.
        
        Args:
        - model_path (str): Path to the saved model state_dict.
        - input_dim (int): Input dimension of the classifier model.
        - hidden_dim (int): Hidden dimension of the classifier model.
        - output_dim (int): Output dimension of the classifier model.
        - num_layers (int): Number of layers in the classifier model.
        - device (torch.device): Torch device. Default is CUDA if available, otherwise CPU.
        
        Returns:
        - ActionClassifier: The loaded model.
        """
        # Instantiate a new ActionClassifier model with the given parameters
        classify_model = ActionClassifier(input_dim, hidden_dim, output_dim, num_layers).to(device)
        # Load the model parameters from the saved state_dict
        classify_model.load_state_dict(torch.load(model_path, map_location=device))
        # Set the model to evaluation mode (this affects layers like Dropout)
        classify_model.eval()
        return classify_model

    def save(self, model_path):
        """Save the current state of the model to the given path.
        
        Args:
        - model_path (str): The path where the model state_dict will be saved.
        """
        torch.save(self.state_dict(), model_path)


# model for decoder
class TrajectoryModel(nn.Module):
    def __init__(self):
        """Initialize the TrajectoryModel model."""
        super(TrajectoryModel, self).__init__()
        # Define the LSTM layer to process input of size 63 and produce output of size 128
        # with 2 LSTM layers, using a batch-first arrangement for the input tensor
        self.lstm = nn.LSTM(input_size=63, hidden_size=128, num_layers=2, batch_first=True)
        # Define a fully connected layer that maps the LSTM output (128) back to the original dimension (63)
        self.fc = nn.Linear(128, 63)

    def forward(self, x):
        """Forward propagation of the neural network.
        
        Args:
        - x (tensor): The input data tensor.
        
        Returns:
        - tensor: The output tensor after processing.
        """
        # Pass the input through the LSTM layer
        out, _ = self.lstm(x)
        # Pass the LSTM output through the fully connected layer
        out = self.fc(out)
        return out
    
    @staticmethod
    def save(models_dict, save_dir="saved_models"):
        """Save multiple models from a dictionary to the specified directory.
        
        Args:
        - models_dict (dict): Dictionary of {label: model} pairs.
        - save_dir (str): Directory to save models. Default is 'saved_models'.
        """
        # Ensure the save directory exists, create if not
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Iterate over models in the dictionary
        for label, model in models_dict.items():
            # Convert the label to its string representation to ensure it's safe for file naming
            safe_label = repr(label)
            # Save each model's state_dict to a .pt file named after its label
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_{safe_label}.pt"))

    @staticmethod
    def load(label_to_encoded, device, save_dir="saved_models"):
        """Load multiple models into a dictionary.
        
        Args:
        - label_to_encoded (dict): Dictionary of labels mapping to encoded data.
        - device (torch.device): Torch device.
        - save_dir (str): Directory from where models are to be loaded. Default is 'saved_models'.
        
        Returns:
        - dict: Dictionary of {label: model} pairs.
        """
        models_dict = {}
        # Iterate over labels in the given dictionary
        for label in label_to_encoded.keys():
            # Convert the label to its string representation to ensure it matches the saved file name
            safe_label = repr(label)
            # Instantiate a new TrajectoryModel and load its state from the saved file
            model = TrajectoryModel().to(device)
            model.load_state_dict(torch.load(os.path.join(save_dir, f"model_{safe_label}.pt")))
            # Add the loaded model to the models dictionary with its label
            models_dict[label] = model
        return models_dict

    @staticmethod
    def generate_trajectory(models_dict, label, start_point, device, sequence_length=330):
        """Generate a trajectory for a given class and initial point.
        
        Args:
        - models_dict (dict): Dictionary of {label: model} pairs.
        - label: Label of the model to use.
        - start_point (array): Initial point for trajectory generation.
        - device (torch.device): Torch device.
        - sequence_length (int): Length of the trajectory sequence. Default is 330.
        
        Returns:
        - np.array: Predicted trajectory reshaped to the format (-1, 21, 3).
        """
        # Get the model for the specified label
        model = models_dict[label]
        # Convert the start point to a tensor, add an extra dimension to simulate a batch, and move it to the specified device
        inputs = torch.tensor(start_point, dtype=torch.float32).unsqueeze(0).to(device)
        predicted_trajectory = [start_point]

        # For each point in the sequence, minus the start point
        for _ in range(sequence_length - 1):
            # Generate a prediction using the model
            output = model(inputs)
            # Add the prediction to the trajectory list and use it as the next input
            predicted_trajectory.append(output.squeeze().cpu().detach().numpy())
            inputs = output

        # Convert the trajectory list to a numpy array and reshape it to the format (-1, 21, 3)
        return np.array(predicted_trajectory).reshape(-1, 21, 3)



