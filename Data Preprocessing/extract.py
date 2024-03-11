import numpy as np
import os
import torch

def load_and_save_data(folder_path, action_file, saved_path, save_format='npy'):
    """
    Load .npy files specified in the action_file, extract 'skel_body0' data, and save it.

    :param folder_path: Path to the folder containing .npy files.
    :param action_file: Path to the text file containing action codes.
    :param save_format: Format to save the extracted data ('npy' or 'torch').
    """
    # Read the action codes from the file
    with open(action_file, 'r') as file:
        actions = [line.split(':')[0].strip() for line in file.readlines()]
    print("Loaded action codes from:", action_file)

    for file_name in os.listdir(folder_path):
        # Check if the file name contains any of the action codes
        if any(action in file_name for action in actions):
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing file: {file_name}")
            data = np.load(file_path, allow_pickle=True).item()
            skel_body0_data = data.get('skel_body0')

            if skel_body0_data is not None:
                # Extract parts of the file name for new naming convention
                parts = file_name.split('A')[0].rstrip('.').split('P')
                new_folder_name = file_name.split('A')[1][:3]  # Extract action code
                new_file_name = 'P'.join(parts)  # Reconstruct file name without action code

                # Create new directory for action code if it doesn't exist
                action_folder_path = saved_path + 'A' + new_folder_name+'/'
                if not os.path.exists(action_folder_path):
                    os.makedirs(action_folder_path)
                    print(f"Created new folder: A {new_folder_name}")

                # Save the file in the new format
                save_path = os.path.join(action_folder_path, new_file_name)
                if save_format == 'npy':
                    np.save(save_path + '.npy', skel_body0_data)
                    print(f"Saved .npy file at: {save_path}.npy")
                elif save_format == 'torch':
                    torch_data = torch.from_numpy(skel_body0_data)
                    torch.save(torch_data, save_path + '.pt')
                    print(f"Saved .pt file at: {save_path}.pt")
        else:
            print(f"Skipped file: {file_name}")

if __name__ == "__main__":
    folder_path = 'C:/Users/randulf/Desktop/data/transferred/'
    action_file = 'C:/Users/randulf/Desktop/data/data preprocessing/full.txt'
    saved_path = 'C:/Users/randulf/Desktop/data/extracted/'
    save_format = 'torch'
    load_and_save_data(folder_path, action_file, saved_path, save_format)