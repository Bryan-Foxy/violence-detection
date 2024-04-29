# Use an official Python runtime as a parent image
FROM python:3.11.5

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies required by h5py
RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    libhdf5-serial-dev \
    libhdf5-103 \
    liblzma-dev \
    libzmq3-dev \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 7860 available to the world outside this container
EXPOSE 7860

# Run app.py when the container launches
CMD ["python", "gradio_interface.py"]
