
# ASL Fingerspelling Recognition with TensorFlow

This project demonstrates how to train a Transformer model using TensorFlow to recognize and translate American Sign Language (ASL) fingerspelling from video frames. It leverages the `Google - ASL Fingerspelling Recognition` dataset.

## Project Overview

The objective of this project is to create a deep learning model that can predict and translate ASL fingerspelling into text. This is accomplished by using a Transformer model architecture trained on video frames containing hands making ASL signs.

## Steps Covered in This Notebook:

### 1. Installation
Before running the project, specific dependencies such as `mediapipe` are installed, which help with processing video frames and extracting hand landmarks.

```bash
pip install mediapipe tensorflow_addons
```

### 2. Importing Libraries
Key libraries used in the project include:

- **TensorFlow and Keras**: For model training and architecture.
- **MediaPipe**: For hand landmark recognition from video frames.
- **Matplotlib, NumPy, and Pandas**: For visualization, data manipulation, and analysis.
- **TensorFlow Addons**: For additional TensorFlow utilities, though TensorFlow Addons is entering minimal maintenance mode.

```python
import tensorflow as tf
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
```

### 3. Data Loading
The dataset used in this project is stored in Parquet files and contains sequences of video frames. These frames are transformed into the necessary format for training.

### 4. Preprocessing
The project transforms the raw data into TFRecords format to make training more efficient. Additionally, landmarks such as hand positions are extracted from the video data using the `mediapipe` library.

### 5. Model Architecture
A Transformer model is used for recognizing the fingerspelling. Transformers are particularly suited for sequence prediction tasks like this one.

### 6. Model Conversion to TFLite
After training the model, it is converted to TensorFlow Lite (TFLite) format for better performance and deployment on mobile or edge devices.

### 7. Submission Creation
The final predictions of the model are written to a submission file for evaluation.

## Usage

To use this notebook for your own project:
1. Download the dataset from the competition or dataset source.
2. Ensure that all required libraries are installed.
3. Run the notebook cells in order to train the model.
4. Convert the model to TFLite format for deployment.

## Dependencies

- Python 3.10+
- TensorFlow 2.6+
- MediaPipe 0.10+
- TensorFlow Addons

## Notes

- The TensorFlow Addons library is being deprecated and will be in minimal maintenance mode by May 2024.
- MediaPipe is used to extract hand landmarks for each frame of video.

## License
This project is licensed under the MIT License.
