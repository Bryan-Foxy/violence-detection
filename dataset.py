import glob
import gc
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import tensorflow as tf

video_path = []
targets = []

for i, cls in tqdm(enumerate(['NonViolence', 'Violence']), desc = 'Load video', ascii = True):
    sub_video_paths = glob.glob(f"../video_data/real life violence situations/Real Life Violence Dataset/{cls}/**.mp4")
    sub_video_paths += glob.glob(f"../video_data/real life violence situations/Real Life Violence Dataset/{cls}/**/*.avi", recursive=True)
    video_path += sub_video_paths
    targets += [i] * len(sub_video_paths)
 
