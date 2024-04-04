import random 
import cv2
import imageio
import numpy as np 
import dataset

import tensorflow as tf
from tensorflow_docs.vis import embed

def format_frames(frame, output_size):
    """
    Create frame and Resize
    """
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    frame = tf.image.resize_with_pad(frame, *output_size)
    return frame

def frame_from_video(video_path, n_frames, output_size = (224,224), frame_step = 15):
    """
    Load Video and make frame 
    """
    result = []
    source = cv2.VideoCapture(str(video_path))
    video_length = source.get(cv2.CAP_PROP_FRAME_COUNT)
    need_length = 1 + (n_frames - 1) * frame_step

    if need_length > video_length:
        start = 0
    else:
        max_start = video_length - need_length
        start = random.randint(0, max_start + 1)
    
    source.set(cv2.CAP_PROP_POS_FRAMES, start)
    ret, frame = source.read()
    result.append(format_frames(frame, output_size))

    for _ in range(n_frames - 1):
        for _ in range(frame_step):
            ret, frame = source.read()
        if ret:
            frame = format_frames(frame, output_size)
            result.append(frame)
        else:
            result.append(np.zeros_like(result[0]))
    source.release()
    result = np.array(result)[..., [2, 1, 0]]

    return result

def to_gif(imgs):
    """
    Frame to gif
    """
    convert_imgs = np.clip(imgs * 255, 0, 255).astype(np.uint8)
    frame_duration = 100 // 10
    imageio.mimsave('saves/gif/animation.gif', convert_imgs, duration = frame_duration)
    return embed.embed_file('saves/gif/animation.gif') 



if __name__ == '__main__':
    # Test code
    sample_video = frame_from_video(dataset.video_path[0], n_frames = 10)
    print(sample_video.shape)
    to_gif(sample_video)
