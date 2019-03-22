import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.validation import check_random_state
from sklearn.datasets import fetch_olivetti_faces
from sklearn.externals import joblib


rng = check_random_state(21)
dataset = fetch_olivetti_faces()
X = dataset.images.reshape(dataset.images.shape[0], -1) 

train = X[dataset.target < 30]
test = X[dataset.target >= 30]
n_faces = 3 
face_ids = rng.randint(test.shape[0], size=(n_faces,))
test = test[face_ids, :]
n_pixels = X.shape[1]

# Upper half of the faces
X_train = train[:, :(n_pixels + 1) // 2]

# Lower half of the faces
y_train = train[:, n_pixels // 2:]
X_test = test[:, :(n_pixels + 1) // 2]
y_test = test[:, n_pixels // 2:]

n_rows = 2
imshape = (64, 64,)


def test_model(y_pred, model_name):
    plt.figure(figsize=(1.7*n_faces, 4))
    plt.suptitle('Face completion with ' + model_name, size=12)
 
    # plot the true faces first
    for i in range(n_faces):
        plt.subplot(int( '{}{}{}'.format( n_rows, n_faces, i + 1 ) ))
        plt.axis('off')
        plt.imshow(np.hstack((X_test[i], y_test[i])).reshape(imshape), cmap=plt.cm.gray, interpolation='nearest')
        

    # then plot the predictions
    for i in range(n_faces):
        plt.subplot(int( '{}{}{}'.format( n_rows, n_faces, i + n_faces + 1 ) ))
        plt.axis('off')
        plt.imshow(np.hstack((X_test[i], y_pred[i])).reshape(imshape), cmap=plt.cm.gray, interpolation='nearest')


test_model(joblib.load('../trained_models/nn_face_completion.pkl').predict(X_test), 'Face completion with a Neural Network')
test_model(joblib.load('../trained_models/knn_face_completion.pkl').predict(X_test), 'Face completion with a k-Nearest Neighbors')
test_model(joblib.load('../trained_models/dt_face_completion.pkl').predict(X_test), 'Face completion with a Decision Tree')
plt.show()
