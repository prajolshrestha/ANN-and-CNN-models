# MNIST-Handwritten-Digit-Recognition-using-TensorFlow

Introduction
This is a project for handwritten digit recognition using TensorFlow, a popular deep learning library. The project uses the MNIST dataset, which consists of 60,000 training images and 10,000 testing images of handwritten digits. The goal of this project is to create a machine learning model that can accurately recognize handwritten digits.

Installation
To run this project, you must have Python 3.10.7 and TensorFlow installed on your computer. 
You can install TensorFlow using pip by running the following command in your command prompt or terminal:
pip install tensorflow

Dataset
The MNIST dataset consists of 60,000 training images and 10,000 testing images of handwritten digits. Each image is 28 x 28 pixels in size and contains a grayscale value between 0 and 255.

Model Creation and Training
The machine learning model used for this project is a sequential model consisting of three layers: 
a flattening layer, 
a dense layer with 128 neurons and a rectified linear unit (ReLU) activation function, 
a dropout layer with a 20% dropout rate, and 
a dense layer with 10 neurons and a softmax activation function. 
The model is compiled with the Adam optimizer, sparse categorical crossentropy loss function, and accuracy as the metric. 
The model is trained on the training images and labels for 25 epochs.( We can choose any number of epoch, but choose wisely) 

Evaluation and Prediction
The model is evaluated on the testing images and labels using the evaluate() method. 
The model's predictions for the testing images are generated using the predict() method, and the index of the maximum value of the predictions for each image is used as the predicted label. 
The model's accuracy is computed using the confusion_matrix() method from the scikit-learn library.

Results
The loss and accuracy of the model during training are plotted using the matplotlib library. A sample misclassified image is shown using matplotlib, along with its true label and predicted label. The confusion matrix is also displayed to show the accuracy of the model's predictions for each digit.


#Credits 
This project was built by PRAJOL SHRESTHA as a personal project. If you have any feedback or suggestions, feel free to create a pull request or contact me via email.

