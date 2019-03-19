# Trained models

## Usage

```py
from sklearn.externals.joblib import load
MODEL_NAME = '...'
model = load(MODEL_NAME)

# For data and the order of data inputs see the trainer's notebook or python file
INPUT_DATA = np.array([
  [FIRST_MEASUREMENT],
  [SECOND_MEASUREMENT],
  .
  .
  .
  [NTH_MEASUREMENT]
])

# Making predictions: 
print(model.predict(INPUT_DATA))
```
