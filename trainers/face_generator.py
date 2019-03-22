import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.utils.validation import check_random_state
from sklearn.externals import joblib

from sklearn.neural_network import MLPRegressor


r = check_random_state(1)
sns.set(rc={ 'figure.figsize': (2, 2) })
data = fetch_olivetti_faces()
imshape = (64, 64,)

X = data.data
y = data.target


# generate faces as a test
def test_model(model=None):
    if not model:
        model = joblib.load('../trained_models/face_generator_nn.pkl')
    
    predictions = model.predict([
        [0],
        [20],
        [40]
    ])

    for img in predictions:
        plt.imshow(img.reshape(*imshape), cmap='gray')
        plt.xticks(())
        plt.yticks(())
        plt.show()


# train an nn as a face generator, so I replace y, and X
print('Training the model ...')
model = MLPRegressor(
        hidden_layer_sizes=(500,), 
        activation='relu', 
        solver='adam', 
        alpha=0.0001,
        batch_size='auto',
        learning_rate='constant', 
        learning_rate_init=0.001,
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
)\
.fit(y.reshape(-1,1), X)
joblib.dump(model, '../trained_models/face_generator_nn.pkl')  # save the model
test_model(model)
