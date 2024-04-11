# Perceptron Classifier for Iris Dataset
This project implements a simple Perceptron algorithm in Python to classify flowers in the Iris dataset into 'Iris-setosa' and 'non-Iris-setosa' categories. It demonstrates the foundational concepts of machine learning, including binary classification, the perceptron learning algorithm, and the concept of epochs in training.

# Features
<br /> Data Loading: Load your dataset from a file with space-separated values, where the last column represents the class label.
<br /> Perceptron Initialization: Initialize perceptron weights to zero, including an additional weight for the bias term.
<br /> Binary Classification: Use the perceptron algorithm to classify data points as 'Iris-setosa' or 'non-Iris-setosa'.
<br /> Perceptron Training: Adjust the perceptron weights using the simple learning rule based on the dataset provided.
<br /> Accuracy Calculation: Evaluate the perceptron's performance on a test set and calculate the accuracy.
<br /> User Interface: A simple user interface to conduct experiments by loading data, setting parameters, and entering new data points for classification.

# Technologies
Python 3

# Experimenting with the Classifier
The user interface prompts for the following:

<br />Path to the training data file.
<br />Path to the test data file.
<br />Learning rate (l_rate).
<br />Number of epochs (n_epoch).
<br />After training, you can input attribute vectors to classify them in real-time or type 'koniec' to quit.

# Dataset Format
The expected dataset format is space-separated values with the last column being the string label of the class. The classifier is binary and will classify labels as either 'Iris-setosa' or 'non-Iris-setosa'.
