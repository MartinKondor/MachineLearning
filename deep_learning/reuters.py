"""
Reuters newswire topics classification

Dataset of 11,228 newswires from Reuters, labeled over 46 topics.
As with the IMDB dataset, each wire is encoded as a sequence
of word indexes (same conventions).
"""
import numpy as np
from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt


# Loading dataset
(xtrain, ytrain), (xtest, ytest) = reuters.load_data(num_words=10000)

# Preprocessing data
def vectorize_x(X):
    ret = np.zeros((len(X), 10000,))
    for i, s in enumerate(X):
        ret[i, s] = 1
    return ret

xtrain = vectorize_x(xtrain)
ytrain = to_categorical(ytrain)

# Building the model
model = Sequential()
model.add(Dense(126, activation='relu', input_shape=(10000,)))
model.add(Dense(46, activation='softmax'))
model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='rmsprop')

# Spliting data to training and validation sets
xval, yval = xtrain[:1000], ytrain[:1000]
xtra, ytra = xtrain[1000:], ytrain[1000:]

# Training the model
n_epochs = 3
h = model.fit(x=xtra, y=ytra, validation_data=(xval, yval,), epochs=n_epochs, batch_size=100)
del xval, yval, xtra, ytra, xtrain, ytrain

# Plotting loss
X = range(1, n_epochs + 1)
plt.title('Loss')
plt.plot(X, h.history['loss'], marker='o', color='blue', label='Training loss')
plt.plot(X, h.history['val_loss'], color='orange', label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.title('Accuracy')
plt.plot(X, h.history['acc'], marker='o', color='blue', label='Training accuracy')
plt.plot(X, h.history['val_acc'], color='orange', label='Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

xtest = vectorize_x(xtest)
pred = np.argmax(model.predict(xtest), axis=1)  # Converting probabilities to class values
print('Accuracy on test set:', np.sum(pred == ytest) / len(ytest))
