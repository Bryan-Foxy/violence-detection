import os 
import cv2
import config
from tqdm import tqdm 
import torch
from torch.utils.data import Dataset, DataLoader

class VideoDataset(Dataset):
    """
    Abstract class representing the dataset.

    Parameters:
    - Dataset(torch.utils.data.Dataset): Inheritance from PyTorch Dataset class.

    Returns:
    None

    The main task of the Dataset class is to return a pair of [input, label] every time it is called.
    We can define functions inside the class to preprocess the data and return it in the format we require.
    """

    def __init__(self, root_dir, dict_labels, mode='train', isplit = True, transform = None):
        self.root_dir = root_dir
        self.data = []
        self.labels = dict_labels
        self.mode = mode
        self.idx = 0
        self.isplit = isplit
        self.transform = transform

        # Select the appropriate directory based on the mode
        data_dir = 'train' if mode == 'train' else 'test'
        file_dir = os.listdir(self.root_dir)

        for i in range(len(file_dir)):
            label = file_dir[i]
            img_dir = os.path.join(self.root_dir, label)

            if self.isplit == True:
                dir = os.listdir(img_dir)
                #check data_dir
                if dir[0] == data_dir:
                    self.idx = 0
                elif dir[1] == data_dir:
                    self.idx = 1

                img_path = os.path.join(img_dir, dir[self.idx])
                img_list = os.listdir(img_path)

                for j in tqdm(range(len(img_list)), desc = f'Loading {label}'):
                    path = os.path.join(img_path, img_list[j])
                    self.data.append([path, label])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = cv2.VideoCapture(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        class_id = self.labels[label]

        # Apply transformations if they are provided
        if self.transform:
            img_tensor = self.transform(img)

        # Channels first
        img_tensor = img_tensor.permute(2, 0, 1)


        return img_tensor, class_id
    
if __name__ == '__main__':
    train_dataset = VideoDataset(config.path, transform = config.transform)
    test_dataset = VideoDataset(transform = config.transform)
    loader_train = DataLoader(train_dataset, batch_size = config.batch_size, shuffle = True)
    loader_test = DataLoader(test_dataset, batch_size = config.batch_size, shuffle = False)