# ANN and CNN models

Introduction
This repository contains the implementation of ANN and Convolutional Neural Network (CNN) models for three popular image datasets: MNIST, Fashion-MNIST, and CIFAR-10.

Dependencies
Python 3.6 or above
TensorFlow 2.0 or above
Keras

Datasets
The following datasets are used in this project:

MNIST: A dataset of 60,000 training images and 10,000 testing images of handwritten digits, with each image being a 28x28 grayscale image.
Fashion-MNIST: A dataset of 60,000 training images and 10,000 testing images of 10 different types of clothing, with each image being a 28x28 grayscale image.
CIFAR-10: A dataset of 50,000 training images and 10,000 testing images of 10 different classes of objects, with each image being a 32x32 color image.

CNN Models
The following CNN models are implemented for each dataset:

MNIST: A simple ANN model consisting of a dense layer.
Fashion-MNIST: A deeper CNN model consisting of multiple convolutional and pooling layers, followed by dense layers.
CIFAR-10: A more complex CNN model consisting of multiple convolutional and pooling layers, dropout layers for regularization, and dense layers.

Results
The following table summarizes the accuracy of each model on the corresponding dataset:

Dataset	Model	Accuracy
MNIST	Simple CNN	98.5%
Fashion-MNIST	Deeper CNN	91.3%
CIFAR-10	Complex CNN	77.2%

Conclusion
This project demonstrates the effectiveness of CNN models for image classification tasks on three popular datasets. Further improvements can be made by experimenting with different hyperparameters and architectures of the CNN models.

#Credits 
This project was built by PRAJOL SHRESTHA as a personal project. If you have any feedback or suggestions, feel free to create a pull request or contact me via email.

