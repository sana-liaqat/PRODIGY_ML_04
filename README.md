# Hand Gesture Recognition by Sana Liaqat
Hand gesture recognition uses computer vision and machine learning to interpret human hand movements for intuitive interaction with digital devices. It has applications in virtual reality, sign language translation, and human-computer interaction.

# Table of Contents
1. Project Overview
2. System Requirements
3. Dataset
4. Model Training
5. Evaluation
6. Acknowledgements

# Project Overview
This project aims to develop a hand gesture recognition system using computer vision and machine learning techniques. The system is capable of recognizing various hand gestures in real-time, which can be used in different applications such as virtual reality, sign language translation, and human-computer interaction.

# System Requirements
1. Python 3.7 or higher
2. OpenCV
3. NumPy
4. TensorFlow or PyTorch
5. Scikit-learn
6. Matplotlib
7. Dataset
   
The dataset should consist of images of hand gestures, organized in subdirectories named after the gesture classes (e.g., thumbs_up, thumbs_down, peace). Ensure that the dataset is placed in the data directory before running the preprocessing script.

# Model Training
The model training process involves:

1. Data Preparation: Load and normalize images, and convert labels to one-hot encoding.
2. Model Architecture: Design a Convolutional Neural Network (CNN) suitable for image classification.
3. Compilation: Configure the model with an optimizer, loss function, and metrics.
4. Training: Fit the model to the training data, adjusting for epochs and batch size.
5. Validation: Evaluate the model on a validation set to ensure generalization.
6. Saving: Save the trained model for future use.

# Evaluation
The performance of the hand gesture recognition model was evaluated using a test dataset. The evaluation metrics include precision, recall, F1-score, and support for each gesture class. The model achieved high performance across all classes, with an overall accuracy of 100.0%.

### Evaluation Metrics

| Gesture       | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| 01_palm       | 1.00      | 1.00   | 1.00     | 366     |
| 02_l          | 1.00      | 1.00   | 1.00     | 392     |
| 03_fist        | 1.00      | 1.00   | 1.00     | 404     |
| 04_fist_moved  | 1.00      | 1.00   | 1.00     | 404     |
| 05_thumb       | 1.00      | 1.00   | 1.00     | 403     |
| 06_index       | 1.00      | 1.00   | 1.00     | 409     |
| 07_ok          | 1.00      | 1.00   | 1.00     | 417     |
| 08_palm_moved  | 1.00      | 1.00   | 1.00     | 410     |
| 09_c          | 1.00      | 1.00   | 1.00     | 418     |
| 10_down       | 1.00      | 1.00   | 1.00     | 377     |

### Overall Performance

- **Accuracy:** 100.0%

The model demonstrates excellent performance with perfect precision, recall, and F1-score across all gesture classes, indicating that it correctly identifies each gesture without any errors.

# Acknowledgements
Thanks to the open-source community for providing the tools and libraries used in this project. Special thanks to the creators of the datasets used for training and evaluation.
