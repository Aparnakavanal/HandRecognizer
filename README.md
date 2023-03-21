# HandRecognizer

This code implements a handwritten digit recognition system using a convolutional neural network (CNN) trained on the MNIST dataset. The system is implemented as a Flask application, which can be accessed through a web browser.

The code loads the MNIST dataset using the Keras library and preprocesses it by normalizing the pixel values and converting the labels to one-hot encoded format. A CNN model is defined and trained on the preprocessed data. The trained model is then loaded and used to predict the digits in user-provided images.

# API

The Flask application has two routes:

   The '/' route renders an HTML template with a canvas element where users can draw digits using their mouse.

   The '/predict' route accepts a POST request with the base64-encoded image data and uses the loaded model to predict the digit in the image. The predicted digit is returned as a JSON response.

The code should work as is, assuming that the required libraries are installed and that the 'mnist.h5' model file is present in the same directory as the code file.
