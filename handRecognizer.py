import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from tensorflow.keras.utils import img_to_array

import cv2
import numpy as np
import base64

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = np.reshape(x_train, (60000, 28, 28, 1))
x_test = np.reshape(x_test, (10000, 28, 28, 1))

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

app = Flask(__name__, template_folder='template')

model = load_model('mnist.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image_data = request.form['imageData']

    img = cv2.imdecode(np.fromstring(base64.b64decode(image_data.split(',')[1]), np.uint8), cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (28, 28))

    img = cv2.bitwise_not(img)

    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    digit = np.argmax(prediction)

    return jsonify({'digit': digit})

if __name__ == '__main__':
    app.run(debug=True)