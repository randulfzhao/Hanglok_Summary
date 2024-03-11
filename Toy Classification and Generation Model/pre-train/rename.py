import os
import subprocess
from util.func import count_files_in_directory

def rename_files(original_name, append_string):
    """
    Renames the given video and its associated CSV file by appending a given string to the original filename.
    
    Parameters:
    - original_name (str): The original filename.
    - append_string (str): The string to append to the filename.
    """
    
    # Define paths for video and csv files
    video_path = os.path.join('vid', original_name)
    csv_path = os.path.join('excel', original_name.split('.')[0] + ".csv")

    # Rename the video
    if os.path.exists(video_path):
        video_new_name_base = original_name.split('.')[0] + "_edited"
        video_new_name = video_new_name_base + "_" + append_string + "." + original_name.split('.')[1]
        video_new_path = os.path.join('vid', video_new_name)
        os.rename(video_path, video_new_path)

    # Rename the csv file
    if os.path.exists(csv_path):
        csv_new_name = video_new_name_base + "_" + append_string + ".csv"
        csv_new_path = os.path.join('excel', csv_new_name)
        os.rename(csv_path, csv_new_path)

def delete_file(file_name):
    """
    Deletes the specified video file and its associated CSV file.
    
    Parameters:
    - file_name (str): The name of the file to delete.
    """
    video_path = os.path.join('vid', file_name)
    csv_path = os.path.join('excel', file_name.split('.')[0] + ".csv")

    # Delete video file
    if os.path.exists(video_path):
        os.remove(video_path)
        print(f"Video '{file_name}' has been deleted.")

    # Delete CSV file
    if os.path.exists(csv_path):
        os.remove(csv_path)
        print(f"CSV file '{file_name.split('.')[0]}.csv' has been deleted.")

def play_video_with_default_player(video_path):
    """
    Plays the specified video using the default video player.
    
    Parameters:
    - video_path (str): The path of the video to play.
    """
    # Determine play method based on OS
    if os.name == 'nt':  # for Windows
        os.startfile(video_path)
    else:
        opener = "open" if sys.platform == "darwin" else "xdg-open"  # for macOS and Linux
        subprocess.call([opener, video_path])

def get_append_string():
    """
    Prompts the user to select or input an append string.
    
    Returns:
    - str: The append string chosen or entered by the user.
    """
    default_values = ["drinkWater", "reachOut", "getPhone"]
    print("Please choose a value to append:")
    for i, value in enumerate(default_values):
        print(f"{i+1}. {value}")
    print(f"{len(default_values)+1}. Custom input")

    choice = int(input("Enter your choice (number):"))
    if 1 <= choice <= len(default_values):
        return default_values[choice-1]
    elif choice == len(default_values) + 1:
        return input("Enter a custom string:")
    else:
        print("Invalid choice. Using default value 'drinkWater'.")
        return "drinkWater"

def main():
    """Main execution function that processes video files in the 'vid' directory."""
    print("Starting main function execution...")
    for video_name in os.listdir('vid'):
        # Only process files without the "_edited" tag
        if "_edited" not in video_name:
            video_path = os.path.join('vid', video_name)
            play_video_with_default_player(video_path)

            print(f"Preview of video {video_name} is completed!")
            option = input("Please choose an action:\n1. Rename file\n2. Delete file\nEnter option (1 or 2):")

            # Rename or delete file based on user's choice
            if option == '1':
                append_string = get_append_string()
                rename_files(video_name, append_string)
            elif option == '2':
                delete_file(video_name)
            else:
                print("Invalid choice!")
    print("All renaming tasks have been completed, ending main function execution. Congrats!")
    count_files_in_directory()  # Note: 'count_files_in_directory()' appears to be missing in the provided code. Ensure it's defined elsewhere in your code.


if __name__ == "__main__":
    main()