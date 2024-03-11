import os
import ast
import torch
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from util.models import ActionClassifier,TrajectoryModel
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from util.func import *



# ---------------------------- Check the Loaded Frame's Shape ----------------------------

def check_shape(data, desired_shape):
    """
    Validates if the input data structure conforms to the desired shape.
    
    :param data: List of actions where each action contains keyframes and each keyframe contains keypoints.
    :param desired_shape: Tuple indicating the desired shape in the format (action_length, keyframe_length, keypoint_length).
    """
    
    # Iterate over each action in the data
    for i, action in enumerate(data):
        
        # Check if the number of keyframes in the current action matches the desired shape
        if len(action) != desired_shape[0]:
            print(f"Action at index {i} has {len(action)} keyframes instead of {desired_shape[0]}.")
            
        # Iterate over each keyframe in the current action
        for j, keyframe in enumerate(action):
            
            # Check if the number of keypoints in the current keyframe matches the desired shape
            if len(keyframe) != desired_shape[1]:
                print(f"Keyframe {j} in action at index {i} has {len(keyframe)} keypoints instead of {desired_shape[1]}.")
                
            # Iterate over each keypoint in the current keyframe
            for k, keypoint in enumerate(keyframe):
                try:
                    # Verify if the current keypoint matches the desired shape
                    if len(keypoint) != desired_shape[2]:
                        print(f"Keypoint {k} in keyframe {j} of action at index {i} has a shape of {len(keypoint)} instead of {desired_shape[2]}.")
                except:
                    print(f"Keypoint {k} in keyframe {j} of action at index {i} is {keyframe} instead of list of length {desired_shape[2]}.")


# ---------------------------- Model Training and Evaluation Functions ----------------------------

def get_wrongly_classified_info(outputs, labels):
    """
    Identifies and returns the indices and predictions of wrongly classified results.
    
    :param outputs: Tensor containing model predictions.
    :param labels: Tensor containing true labels.
    :return: Tuple containing lists of wrong indices and their corresponding wrong predictions.
    """
    
    # Determine the predicted classes by taking the maximum value index for each output
    _, predicted = torch.max(outputs.data, 1)
    
    # Identify the indices where predictions don't match with actual labels
    wrong_indices = (predicted != labels).nonzero(as_tuple=True)[0]
    
    # Extract the wrong predictions using these indices
    wrong_predictions = predicted[wrong_indices]
    
    return wrong_indices.tolist(), wrong_predictions.tolist()

# 分割训练集和测试集
def split_data_for_training(data, labels, test_size=0.1):
    """
    Splits the dataset into training and validation subsets.
    
    :param data: List of data points.
    :param labels: List of corresponding labels for each data point.
    :param test_size: Proportion of the dataset to be included in the validation split.
    :return: Tuple containing lists of training data, validation data, training labels, and validation labels.
    """
    
    # Use train_test_split to segment data and labels into training and validation sets
    train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=test_size)
    return train_data, val_data, train_labels, val_labels

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, device, early_stop_patience=10):
    """
    Trains a PyTorch model using the given data loaders, criterion, and optimizer, with early stopping.
    
    :param model: The neural network model to be trained.
    :param criterion: The loss function used for training.
    :param optimizer: The optimization algorithm.
    :param train_loader: DataLoader for the training set.
    :param val_loader: DataLoader for the validation set.
    :param num_epochs: Number of epochs to train for.
    :param device: Device ("cuda" or "cpu") to which tensors will be moved before computation.
    :param early_stop_patience: Number of epochs with no improvement on validation loss to trigger early stopping.
    :return: The best trained model based on validation loss.
    """
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model = None

    # Loop through all epochs
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            # Move tensors to the specified device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward propagation
            outputs = model(inputs)
            
            # Compute loss
            loss = criterion(outputs, labels)
            
            # Backward propagation and optimization
            loss.backward()
            optimizer.step()
            
            # Accumulate training loss
            train_loss += loss.item() * inputs.size(0)
        
        # Compute average training loss
        train_loss /= len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

        # Compute average validation loss
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch}/{num_epochs - 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Check for early stopping based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter == early_stop_patience:
                print("Early stopping triggered.")
                break

    # Load weights of the best model
    model.load_state_dict(best_model)
    return model

def evaluate_model(model, test_loader, device):
    """
    Evaluates the performance of a trained model on a test dataset.
    
    :param model: The trained neural network model.
    :param test_loader: DataLoader for the test set.
    :param device: Device ("cuda" or "cpu") to which tensors will be moved before computation.
    :return: The accuracy of the model on the test set (as a percentage).
    """
    
    # Move the model to the specified device
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Predict class labels
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            
            # Count number of correct predictions
            correct += (predicted == labels).sum().item()

    # Return accuracy as a percentage
    return 100 * correct / total


# ---------------------------- Classifier Training & Testing Functions -----------------------------

def train_classifier(data, labels, classify_model_path):
    """
    Trains the classifier using the provided data and saves the trained model.
    
    :param data: The dataset to be used for training.
    :param labels: Labels for the dataset.
    :param classify_model_path: Path where the trained model should be saved.
    :return: Trained classifier model and keyframe.
    """

    # Split data into training and validation sets
    train_data, val_data, train_labels, val_labels = split_data_for_training(data, labels)

    # Convert data to PyTorch tensors
    train_data_tensor = torch.tensor(train_data, dtype=torch.float32)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
    val_data_tensor = torch.tensor(val_data, dtype=torch.float32)
    val_labels_tensor = torch.tensor(val_labels, dtype=torch.long)

    # Create PyTorch data loaders for training and validation datasets
    train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = TensorDataset(val_data_tensor, val_labels_tensor)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model, loss function, and optimizer
    classify_model = ActionClassifier(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classify_model.parameters(), lr=LEARNING_RATE)

    # Train the classifier model
    classify_model = train_model(classify_model, criterion, optimizer, train_loader, val_loader, NUM_EPOCHS, device, EARLY_STOP_PATIENCE)

    # Save the trained classifier model to the specified path
    classify_model.save(model_path=classify_model_path)
    print("Classifier Saved")

    return classify_model, keyframe

def test_classifier(data, labels, keyframe, classify_model, device):
    """
    Evaluates the trained classifier on the test data.
    
    :param data: Test data.
    :param labels: Test data labels.
    :param keyframe: Keyframe information.
    :param classify_model: The trained classifier model.
    :param device: The computation device ("cuda" or "cpu").
    """
    
    # Convert test data to PyTorch tensors
    test_data_tensor = torch.tensor(data, dtype=torch.float32)
    test_labels_tensor = torch.tensor(labels, dtype=torch.long)

    # Create PyTorch data loader for test dataset
    test_dataset = TensorDataset(test_data_tensor, test_labels_tensor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Evaluate the classifier model on test data
    accuracy = evaluate_model(classify_model, test_loader, device)
    print(f'Accuracy on the test data: {accuracy:.2f}%')

# ---------------------------- Decoder Training -----------------------------

def decoder(unique_labels, device, data_transformed):
    """
    Trains decoders for each unique label.
    
    :param unique_labels: A list of unique labels.
    :param device: The computation device ("cuda" or "cpu").
    :param data_transformed: Transformed data for training.
    :return: Dictionary containing trained models for each label.
    """
    
    # Initialize models and optimizers for each label
    models_dict = {label: TrajectoryModel().to(device) for label in unique_labels}
    optimizers_dict = {label: optim.Adam(models_dict[label].parameters(), lr=0.001) for label in unique_labels}
    criterion = nn.MSELoss().to(device)

    for epoch in range(100):
        for label in unique_labels:
            loss_list = []  # Initialize loss list for each label
            for i, data_label in enumerate(encoded_labels):
                if data_label == label:
                    model = models_dict[label]
                    optimizer = optimizers_dict[label]
                    
                    # Prepare input and target tensors
                    inputs = torch.tensor(data_transformed[i][:-1], dtype=torch.float32).unsqueeze(0).to(device)
                    targets = torch.tensor(data_transformed[i][1:], dtype=torch.float32).unsqueeze(0).to(device)
                    
                    # Forward propagation
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    # Backward propagation and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    loss_list.append(loss.item())

            # Compute average loss for the current label
            avg_loss = sum(loss_list) / len(loss_list) if loss_list else 0
            print(f"Epoch {epoch + 1} - Label {label}, Avg Loss: {avg_loss:.4f}")

    # Save models after training
    TrajectoryModel.save(models_dict=models_dict)
    
    return models_dict

# ---------------------------- Main Function & Model Execution -----------------------------

if __name__ == "__main__":
    # ---------------- Constants Initialization ----------------
    # Dimensionality of input data: Assuming 21 keypoints with 3D coordinates (x, y, z)
    INPUT_DIM = 21 * 3
    # Number of hidden units in the neural network
    HIDDEN_DIM = 128
    # Number of recurrent layers in the model
    NUM_LAYERS = 2
    # Number of training epochs
    NUM_EPOCHS = 50
    # Size of mini-batches for training
    BATCH_SIZE = 64
    # Output dimension, which will be set later based on the number of unique labels
    OUTPUT_DIM = None
    # Maximum number of consecutive epochs with no improvement in validation loss for early stopping
    EARLY_STOP_PATIENCE = 10
    # Learning rate for optimizer
    LEARNING_RATE = 0.001

    # ---------------- Setting Up Computation Device ----------------
    # If GPU is available, use it; otherwise, fall back to CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---------------- Data Preprocessing ----------------
    # Directory where training data is located
    train_directory = 'excel'
    # Load and preprocess the training data and labels
    train_data, labels, keyframe = preprocess_train(train_directory)
    # Augment the training data 15 times for better training performance
    train_data, labels = augment_data_and_labels(train_data, labels, times=15)
    # Reshape the training data for the decoder. Assuming 63 as 21 keypoints with 3D coordinates
    num_key_frame = len(train_data[0])
    data_transformed = np.array(train_data).reshape(len(train_data), num_key_frame, 63)

    # ---------------- Label Encoding ----------------
    # Get unique labels in the dataset
    unique_labels = set(labels)
    # Create a mapping from unique label to a unique integer (encoding)
    label_to_encoded = {label: i for i, label in enumerate(unique_labels)}
    # Inverse of the above mapping
    encoded_to_label = {i: label for label, i in label_to_encoded.items()}
    # Setting the output dimension to the number of unique labels/classes
    OUTPUT_DIM = len(label_to_encoded)
    print(label_to_encoded)
    # Convert the actual labels to their encoded counterparts for model training
    encoded_labels = [label_to_encoded[l] for l in labels]
    # Save the mappings for future reference or use
    save_label_dicts(label_to_encoded, encoded_to_label)
    # Uncomment the below line if you wish to load the saved mappings
    # loaded_label_to_encoded, loaded_encoded_to_label = load_label_dicts()

    # ---------------- Train Classifier Model ----------------
    # Path to save the trained classifier model
    classify_model_path = "saved_models/model_checkpoint.pth"
    # Train the classifier and get the resulting model
    classify_model, keyframe = train_classifier(train_data, encoded_labels, classify_model_path)

    # ---------------- Test the Classifier Model ----------------
    # Directory where test data is located
    test_directory = 'data_test'
    # Load and preprocess the test data
    test_data, test_labels = preprocess_test(test_directory, keyframe)
    # Convert the test labels to their encoded form
    encoded_test_labels = [label_to_encoded[label] for label in test_labels]
    # Evaluate the trained classifier on the test dataset
    test_classifier(test_data, encoded_test_labels, keyframe, classify_model, device)

    # ---------------- Train Encoder/Decoder Models ----------------
    # Train separate encoder/decoder models for each unique label and get the resulting dictionary of models
    models_dict = decoder(unique_labels, device, data_transformed)

    # ---------------- Generate Trajectory for a Given Label ----------------
    # Label for which trajectory is to be generated. Change this as needed.
    trajectory_label = 0  
    # Generate the trajectory for the given label using the trained models
    trajectory = TrajectoryModel.generate_trajectory(models_dict, encoded_to_label[trajectory_label], data_transformed[0][0], device)
    print(trajectory)
