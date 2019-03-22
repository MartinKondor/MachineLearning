import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils.validation import check_random_state
from sklearn.externals import joblib
from random import randint

from sklearn.neural_network import MLPRegressor


r = check_random_state(1)
sns.set(rc={ 'figure.figsize': (2, 2) })
imshape = (64, 64,)


# generate faces as a test
def test_model(model=None):
    if not model:
        model = joblib.load('../trained_models/face_generator_nn.pkl')
    
    predictions = model.predict([
        [randint(0, 1000)],
    ])

    for img in predictions:
        plt.imshow(img.reshape(*imshape), cmap='gray')
        plt.xticks(())
        plt.yticks(())
        plt.show()


test_model()
