import torchvision
#import torchvision.transforms._transforms_video as V

path = '../video_data/Real Life Violence Dataset'
batch_size = 16
epochs = 100
lr = 2e-4

"""
# Transformation
transform = torchvision.transforms.Compose([
    V.RandomCropVideo(),
    V.NormalizeVideo(mean = [1,1,1], std = [0,0,0]),
])
"""