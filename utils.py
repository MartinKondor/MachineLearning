
def accuracy(y, y_hat):
    good = 0
    
    for i in range(y_hat.shape[0]):
        if y[i] == 1 and y_hat[i] > .5:
            good += 1
        elif y[i] == 0 and y_hat[i] <= .5:
            good += 1 
            
    return 100*good / y_hat.shape[0]


# Sum all the columns
def sumcols(df):
    """
    :param df: pandas.DataFrame
    """
    X = df.loc[:, df.columns[0]].values

    for i in range(1, len(df.columns)):
        X += df.loc[:, df.columns[i]].values

    return X
    
    
def get_regr_coeffs(X, y):
    n = X.shape[0]
    p = n*(X**2).sum() - X.sum()**2
    a = ( n*(X*y).sum() - X.sum()*y.sum() ) / p
    b = ( y.sum()*(X**2).sum() - X.sum()*(X*y).sum() ) / p
    return a,b
    
