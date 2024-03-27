# Importing os module
import os
import shutil
import config
from tqdm import tqdm 

PATH = config.path

def split_dataset(path, train_data=0.8):
    """
    Split the dataset into a train and test set.

    Parameters:
    - path (str): The main directory containing subdirectories for each label.
    - train_data (float): Percentage of samples for the train set. Default is 0.8 (80%).

    Returns:
    None

    This function takes the input path and divides the dataset into a train set and a test set.
    Two folders, 'train' and 'test', will be created if they do not exist, and the specified percentage of samples with file extensions .jpg, .jpeg, or .png will be moved to the 'train' folder.
    If there are multiple subfolders, this operation will be performed for each subfolder.
    """

    file_path = os.listdir(path)
    train = 'train'
    test = 'test'

    for file in file_path:
      directory = os.path.join(path, file)

      # Check if 'train' and 'test' folders exist; if not, create them
      train_folder = os.path.join(directory, train)
      test_folder = os.path.join(directory, test)
      if not os.path.exists(train_folder):
        os.mkdir(train_folder)
      if not os.path.exists(test_folder):
        os.mkdir(test_folder)

      total_files = len(os.listdir(directory))
      if total_files == 0:
        print(f"No files found in the directory: {directory}")
        continue

      test_data = 1.0 - train_data
      percent = round(total_files * train_data)

      hero = 0  # Variable to help skip files without valid extensions
      print(file)

      for i in tqdm(range(percent), desc = f'Load train {file}'):
        while hero < total_files:
          img_path = os.listdir(directory)[0 + hero]
          if img_path.lower().endswith(('.jpg', '.jpeg', '.png', '.mp4', '.avi')):
              shutil.move(os.path.join(directory, img_path), os.path.join(directory, train))
              break  # exit the loop once a valid file is found
          else:
            hero += 1

          # Break out of the loop if there are no more valid files
          if hero == total_files:
            break

      for j in tqdm(range(total_files - percent), desc = f'Load test {file}'):
        while hero < total_files:
          img_path = os.listdir(directory)[0 + hero]
          if img_path.lower().endswith(('.jpg', '.jpeg', '.png', '.mp4', '.avi')):
            shutil.move(os.path.join(directory, img_path), os.path.join(directory, test))
            break  # exit the loop once a valid file is found
          else:
            hero += 1

          # Break out of the loop if there are no more valid files
          if hero == total_files:
            break

split_dataset(PATH)
print('Operation finished')

