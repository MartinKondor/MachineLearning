

def get_regr_coeffs(X, y):
    n = X.shape[0]
    p = n*(X**2).sum() - X.sum()**2
    a = ( n*(X*y).sum() - X.sum()*y.sum() ) / p
    b = ( y.sum()*(X**2).sum() - X.sum()*(X*y).sum() ) / p
    return a,b
    
