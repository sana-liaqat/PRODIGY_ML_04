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
The model can be trained using either TensorFlow or PyTorch. The training script train.py handles the training process, including data loading, model training, and saving the trained model.

import numpy as np import tensorflow as tf from tensorflow.keras.models import Sequential from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

img_size = 64 num_classes = 2

Load preprocessed data
images = np.load('images.npy') labels = np.load('labels.npy')

Normalize the images
images = images / 255.0

Convert labels to one-hot encoding
labels = tf.keras.utils.to_categorical(labels, num_classes)

Create a simple CNN model
model = Sequential([ Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)), MaxPooling2D((2, 2)), Flatten(), Dense(128, activation='relu'), Dense(num_classes, activation='softmax') ])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

Train the model
model.fit(images, labels, epochs=10, batch_size=32, validation_split=0.2)

Save the model
model.save('hand_gesture_model.h5')

# Evaluation

The model can be evaluated using the evaluate.py script. This script loads the trained model and evaluates its performance on the test dataset, providing metrics such as accuracy, precision, recall, and F1-score.

           precision    recall  f1-score   support

  01_palm       1.00      1.00      1.00       366
     02_l       1.00      1.00      1.00       392
  03_fist       1.00      1.00      1.00       404

04_fist_moved 1.00 1.00 1.00 404 05_thumb 1.00 1.00 1.00 403 06_index 1.00 1.00 1.00 409 07_ok 1.00 1.00 1.00 417 08_palm_moved 1.00 1.00 1.00 410 09_c 1.00 1.00 1.00 418 10_down 1.00 1.00 1.00 377

accuracy                            1.00      4000
macro avg       1.00      1.00      1.00      4000
weighted avg 1.00 1.00 1.00 4000

# Accuracy of the Model: 100.0%

# Acknowledgements
Thanks to the open-source community for providing the tools and libraries used in this project. Special thanks to the creators of the datasets used for training and evaluation.
