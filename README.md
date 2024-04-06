# Video Classification

## Introduction

In this project, we have developed an interesting model that combines Convolution 3D and LSTM to capture spatial and temporal information in videos.

The objective of our work here is to efficiently and accurately identify violent and non-violent videos using the neural network.

**Model Weights**

The model weights are not available in the GitHub repository. To access them, please visit [this Google Drive link](https://drive.google.com/drive/folders/1Ar8gs1rSR7QXci-kcJ_u8yFuujqQuump?usp=sharing).

## Dependencies and Installation
- Python >= 3.7 (We recommend using [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [TensorFlow >= 2.x](https://www.tensorflow.org/?hl=fr)

### Installation

```bash
pip install -r requirements.txt
```

After that, download the TensorFlow documentation:

```bash
pip install -q git+https://github.com/tensorflow/docs
```

**Next, download this repository from our GitHub:**

```bash
git clone https://github.com/Bryan-Foxy/violence-detection/tree/main
cd violence-detection-main
```

----
## Inference
The `inference.py` file is used to test the pre-trained model. Simply change the video path in the code to your own video, and you're good to go.

### How it Works
After preparing the videos by dividing them into several frames and batching them into 32, we directly pass the data into a model to capture the spatial and temporal information of the videos. In the last neuron output, we use `sigmoid` because the problem here is binary (either Violent or Not Violent).

### Testing
Here, we load the test videos and evaluate them with the training data. To visualize the model output, we have implemented a function to load the video path as input, prepare it like the training data, inject it into the model, and then put the obtained output into a box (green if non-violent and red otherwise) on the video for better visibility and interpretation of the model.

Here is an example after testing our model:
<p>
    <img src="saves/gif/archer-_NV_.gif">
    <img src="saves/gif/friends-_NV_.gif">
    <img src="saves/gif/violent-guy.gif">
</p>

