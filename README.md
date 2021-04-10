# LIBRAS-Image-Classifier

This project demonstrates the use of neural networks and computer vision to create a classifier that interprets the Brazilian Sign Language. At the moment, the project interprets only the first 6 letters of the alphabet.
A Convolutional Neural Network was used in order to train and test a network so that, through the webcam, it is possible to identify the signal made by the user's hand.

![libras](https://user-images.githubusercontent.com/31252524/114275902-abe94f80-99fa-11eb-9cfe-fe2cf75269f9.gif)

## Data prediction

In order to teach the neural network, we obtain image captures of the hand signal with the webcam inported in grayscale on the /data folter. There are a total of 1200 images, 200 of each signal.

![image](https://user-images.githubusercontent.com/31252524/114276105-99234a80-99fb-11eb-9fb9-2e9fd3801dd7.png)

## Technologies used

- Keras
- Tensorflow
- OpenCV
- Python 3.8
