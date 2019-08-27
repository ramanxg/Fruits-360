# Fruits-Classification

From Fruits-360 dataset on Kaggle.
Created a Convolutional Neural Network with 6 Convolutional Layers and 3 Dense layers. Achieved 97.4% accuracy on the test set. 

Used Tensorflow with Keras to construct model, and Sci-kit learn and OpenCV to preprocess images. 

# API
Deployed model into a Flask API using Flask-RESTful. GET takes in a base64 string of the image and gives a classification and probability.
