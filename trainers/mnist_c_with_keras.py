import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils


np.random.seed(7)

# data preprocessing
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# flatten the 28*28 images to a 784 vector
n_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], n_pixels).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], n_pixels).astype('float32') / 255

# one-hot encode the outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
n_classes = y_test.shape[1]


# creating the model
model = Sequential()
model.add(Dense(n_pixels, input_dim=n_pixels, kernel_initializer='normal', activation='relu'))
model.add(Dense(n_classes, kernel_initializer='normal', activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# training the model
model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=200,
    verbose=2
)
scores = model.evaluate(X_test, y_test, verbose=0)

print('Error:', 100 - scores[1] * 100)
