import dataset
import functions_tools

if __name__ == '__main__':
    # Test code
    video_paths, targets = dataset.load_video_path()
    sample_video = functions_tools.frame_from_video(video_paths[944], n_frames = 10)
    print(sample_video.shape)
    print(f"The class is {targets[944]}: {'NonViolence' if targets[0] == 0 else 'Violence'}")
    functions_tools.to_gif(sample_video)