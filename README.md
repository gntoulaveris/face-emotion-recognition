# face-emotion-recognition

# Overview
This project involved the recognition of face emotions taken from real time image footage from the user's camera. The Keras library was utilized and the model was built in Google Colab to utilize a GPU for faster training. The deployed model allows the user to capture an image from his camera, detects the user's face using haarcascade and assigns to it the appropriate emotion. The model is able to detect with good accuracy between the following emotions: ’anger’, ’disgust’, ’fear’, ’happiness’, ’sadness’, ’surprise’ and ’neutral’.

# Training data
The training dataset was derived from Kaggle: https://www.kaggle.com/code/gauravsharma99/facial-emotion-recognition/data
It composed of 35.887 face images, split between the emotion categories, as seen in the following plot. Some of the face images belonged to cartoons or anime, but the majority seemed to be of real people. An overview of some of them can be seen in the gray-scaled image below. Provided were also a validation and a test set that shared the same classes' imbalance as the training set.

<p align="center">
  <img src="face_emotion_recognition.png" alt="Emotion classes" width="500">
</p>

<p align="center">
  <img src="training_set_faces.png" alt="Faces" width="1000">
</p>

