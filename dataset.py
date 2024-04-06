import numpy as np 
import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import functions_tools
import config

import tensorflow as tf


def load_video_path():
    # Load video_path and return a list and the target class
    video_paths = []
    targets = []

    for i, cls in tqdm(enumerate(['NonViolence', 'Violence']), desc = 'Load video', ascii = True):
        sub_video_paths = glob.glob(f"../video_data/real life violence situations/Real Life Violence Dataset/{cls}/**.mp4")
        sub_video_paths += glob.glob(f"../video_data/real life violence situations/Real Life Violence Dataset/{cls}/**/*.avi", recursive=True)
        video_paths += sub_video_paths
        targets += [i] * len(sub_video_paths)
    
    return video_paths[0:100], targets[0:100]

def load_video_and_transform(video_paths):
    # Load video and transform it in frame for manipulation
    video_dataset = []
    for video_path in tqdm(video_paths):
        video_dataset.append(functions_tools.frame_from_video(video_path, n_frames = 10))

    return np.array(video_dataset) 

def split_data(dataset, target):
    # Split in train and test 
    train_dataset, test_dataset, train_y_dataset, test_y_dataset = train_test_split(dataset, target,
                                                                                    test_size = .2, random_state = 42)
    return train_dataset, test_dataset, train_y_dataset, test_y_dataset

def create_loader(train_dataset, test_dataset, train_y_dataset, test_y_dataset):
    # Create and inject data in dataloader
    train_loader = tf.data.Dataset.from_tensor_slices((train_dataset, train_y_dataset)).shuffle(config.batch * 4).batch(config.batch).cache().prefetch(tf.data.AUTOTUNE)
    test_loader = tf.data.Dataset.from_tensor_slices((test_dataset, test_y_dataset)).batch(config.batch).cache().prefetch(tf.data.AUTOTUNE)

    return train_loader, test_loader

if __name__ == '__main__':
    video_paths, targets = load_video_path()
    video_dataset = load_video_and_transform(video_paths)
    train_dataset, test_dataset, train_y_dataset, test_y_dataset = split_data(video_dataset, targets)
    train_loader, test_loader = create_loader(train_dataset, test_dataset, train_y_dataset, test_y_dataset)
    print('Datas are ready!')
    for X,y in train_loader.take(1):
        print(X.shape)
        print(y.shape)

