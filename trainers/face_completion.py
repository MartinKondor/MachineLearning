import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error as mse, mean_absolute_error as mae
from sklearn.datasets import fetch_olivetti_faces
from sklearn.utils.validation import check_random_state
from sklearn.metrics import mean_squared_error as mse
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor


rng = check_random_state(21)
dataset = fetch_olivetti_faces()
X = dataset.images.reshape(dataset.images.shape[0], -1)  # == dataset.data

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


# Models
print('start training MLPR')
nn_model = MLPRegressor(
        hidden_layer_sizes=(500,), 
        activation='relu', 
        solver='adam', 
        alpha=0.0001,
        batch_size='auto',
        learning_rate='constant', 
        learning_rate_init=0.01,
        power_t=0.001,
        max_iter=2000,
        shuffle=False, 
        random_state=3,
        tol=0.0001, 
        momentum=0.8, 
        nesterovs_momentum=True,
        early_stopping=False,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=0.000000001,
).fit(X_train, y_train)
print('NN Error on train:', mse(y_train, nn_model.predict(X_train)))
print('NN Error on test:', mse(y_test, nn_model.predict(X_test)))
print()

print('start training KNN')
knn_model = KNeighborsRegressor(
    n_neighbors=47,
    weights='uniform',
    algorithm='auto',
    leaf_size=110,
    p=2,
    metric='minkowski',
    n_jobs=-1,
).fit(X_train, y_train)
print('kNN Error on train:', mse(y_train, knn_model.predict(X_train)))
print('kNN Error on test:', mse(y_test, knn_model.predict(X_test)))
print()

print('start training DTR')
dt_model = DecisionTreeRegressor(
    criterion='friedman_mse',
    splitter='best',
    max_depth=500,
    min_samples_split=50,
    min_samples_leaf=100,
    min_weight_fraction_leaf=0.1,
    max_features=1000,
    random_state=0,
    max_leaf_nodes=None,
    min_impurity_decrease=0.00001,
    min_impurity_split=None,
    presort=True,
).fit(X_train, y_train)
print('DT Error on train:', mse(y_train, dt_model.predict(X_train)))
print('DT Error on test:', mse(y_test, dt_model.predict(X_test)))
print()

# Plotting
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


print('save models')
joblib.dump(nn_model, '../trained_models/temp_nn_face_completion.pkl')
joblib.dump(knn_model, '../trained_models/temp_knn_face_completion.pkl')
joblib.dump(dt_model, '../trained_models/temp_dt_face_completion.pkl')

print('testing models')
test_model(nn_model.predict(X_test), 'Face completion with a Neural Network')
test_model(knn_model.predict(X_test), 'Face completion with a k-Nearest Neighbors')
test_model(dt_model.predict(X_test), 'Face completion with a Decision Tree')
plt.show()
