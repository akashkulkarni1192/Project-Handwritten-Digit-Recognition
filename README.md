Project-Handwritten-Digit-Recognition
=====================================
Using MNIST dataset, recognizing digits through different ANN Models

Goal
----
Prediction of digits using a set of 28x28 sized images of handwritten digits from Kaggle's Digit Recognizer dataset i.e MNIST dataset

I solved the problem and applied the optimization techniques in following ways : 
* **Multi-class logistic regression**
* **ANN with stochastic gradient descent algorithm**
* **ANN with batch gradient descent algorithm** : Also implemented in theano.
* **ANN with batch gradient descent algorithm and momentum**
* **ANN with batch gradient descent algorithm and RMSProp** : Implemented in tensorflow

I observed that with each subsequent model, there was an improve in accuracy and convergence time.

### Models to try ###
* **ANN with batch gradient descent algorithm, nesterov momentum and RMSProp**
* **CNN with few convolution-pooling layers on top of ANN model using LeNet architecture**
